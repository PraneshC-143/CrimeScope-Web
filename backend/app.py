import pandas as pd
import numpy as np
import os
import threading
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    has_sklearn = True
except ImportError:
    has_sklearn = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FRONTEND_ROOT = os.path.join(PROJECT_ROOT, "CrimeScope-Web")

app = Flask(__name__)
# Enable CORS so the HTML frontend can query the Python API seamlessly
CORS(app)

# Global dataset cache
_DATA_CACHE = None
_PROJECTION_CACHE = {}
_PROJECTION_RECORD_CACHE = {}
_PROJECTION_WARMING = set()

DEFAULT_PROJECTION_END_YEAR = 2025
PROJECTION_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
OFFICIAL_2023_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "official-crime-data-2023.csv"),
    os.path.join(PROJECT_ROOT, "official-crime-data-2023.xlsx"),
    os.path.join(PROJECT_ROOT, "data", "official", "official-crime-data-2023.csv"),
    os.path.join(PROJECT_ROOT, "data", "official", "official-crime-data-2023.xlsx"),
]


def projection_cache_file(end_year):
    return os.path.join(PROJECTION_CACHE_DIR, f"projections_dataset_{int(end_year)}.json")


def weighted_mean(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    weights = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.average(arr, weights=weights))


def project_next_value(history):
    """Project the next value from a short yearly history with mild damping."""
    arr = np.asarray(history, dtype=float)
    if len(arr) == 0:
        return 0
    if len(arr) == 1:
        return int(max(0, round(arr[-1])))

    recent = arr[-min(4, len(arr)):]
    x_recent = np.arange(len(recent), dtype=float)
    slope = float(np.polyfit(x_recent, recent, 1)[0]) if len(recent) >= 2 else 0.0
    diffs = np.diff(recent)
    momentum = weighted_mean(diffs) if len(diffs) else 0.0
    baseline = weighted_mean(recent[-min(3, len(recent)):])
    last_val = float(recent[-1])

    raw_next = baseline + (0.55 * slope) + (0.45 * momentum)
    max_step = max(abs(last_val) * 0.22, 3.0)
    clipped_next = float(np.clip(raw_next, last_val - max_step, last_val + max_step))
    return int(max(0, round(clipped_next)))


def project_series(history, start_year, end_year):
    projected = []
    series = list(np.asarray(history, dtype=float))
    for year in range(start_year, end_year + 1):
        next_val = project_next_value(series)
        projected.append((year, next_val))
        series.append(next_val)
    return projected


def build_projection_rows(df, crime_cols, end_year=DEFAULT_PROJECTION_END_YEAR):
    cache_key = int(end_year)
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key]

    latest_year = int(df["year"].max())
    if end_year <= latest_year:
        _PROJECTION_CACHE[cache_key] = pd.DataFrame(columns=df.columns.tolist() + ["is_projected", "data_stage"])
        return _PROJECTION_CACHE[cache_key]

    rows = []
    grouped = df.sort_values(["state_name", "district_name", "year"]).groupby(["state_name", "district_name"], sort=False)

    for (state_name, district_name), group in grouped:
        group = group.sort_values("year")
        projected_by_col = {}
        for col in crime_cols:
            projected_by_col[col] = dict(project_series(group[col].fillna(0).values, latest_year + 1, end_year))

        for year in range(latest_year + 1, end_year + 1):
            row = {
                "state_name": state_name,
                "district_name": district_name,
                "year": year,
                "is_projected": True,
                "data_stage": "forecast",
            }
            for col in crime_cols:
                row[col] = projected_by_col[col][year]
            row["total_crimes"] = int(sum(row[col] for col in crime_cols))
            rows.append(row)

    projection_df = pd.DataFrame(rows)
    _PROJECTION_CACHE[cache_key] = projection_df
    return projection_df


def build_projection_records(df, crime_cols, end_year=DEFAULT_PROJECTION_END_YEAR):
    cache_key = int(end_year)
    if cache_key in _PROJECTION_RECORD_CACHE:
        return _PROJECTION_RECORD_CACHE[cache_key]

    cache_path = projection_cache_file(end_year)
    if os.path.exists(cache_path):
        try:
            cached = pd.read_json(cache_path).to_dict(orient='records')
            _PROJECTION_RECORD_CACHE[cache_key] = cached
            return cached
        except Exception:
            pass

    projection_df = build_projection_rows(df, crime_cols, end_year=end_year).copy()
    if projection_df.empty:
        _PROJECTION_RECORD_CACHE[cache_key] = []
        return _PROJECTION_RECORD_CACHE[cache_key]

    rename_map = {
        'state_name': 'State',
        'district_name': 'District',
        'year': 'Year',
        'total_crimes': 'Total',
        'is_projected': 'IsProjected',
        'data_stage': 'DataStage',
    }

    projection_df.rename(columns=rename_map, inplace=True)
    projection_df.fillna(0, inplace=True)
    records = projection_df.to_dict(orient='records')
    try:
        os.makedirs(PROJECTION_CACHE_DIR, exist_ok=True)
        pd.DataFrame(records).to_json(cache_path, orient='records')
    except Exception:
        pass
    _PROJECTION_RECORD_CACHE[cache_key] = records
    return records


def warm_projection_cache_async(df, crime_cols, end_year=DEFAULT_PROJECTION_END_YEAR):
    cache_key = int(end_year)
    if cache_key in _PROJECTION_RECORD_CACHE or cache_key in _PROJECTION_WARMING:
        return

    def runner():
        try:
            build_projection_records(df, crime_cols, end_year=end_year)
        finally:
            _PROJECTION_WARMING.discard(cache_key)

    _PROJECTION_WARMING.add(cache_key)
    threading.Thread(target=runner, daemon=True).start()


def build_features(years):
    years = np.asarray(years, dtype=float)
    min_year = float(years.min())
    return np.column_stack([
        years,
        years - min_year,
        (years - min_year) ** 2,
    ])


def build_prediction_payload(years, totals, target_year):
    years = np.asarray(years, dtype=int)
    totals = np.asarray(totals, dtype=float)

    if len(years) == 0:
        return {
            "target_year": target_year,
            "predicted_crimes": 0,
            "model_used": "Unavailable",
            "confidence_low": 0,
            "confidence_high": 0,
            "delta_from_last_year": 0,
            "pct_change_from_last_year": 0.0,
            "ensemble_components": {},
            "historical_context": {"last_year": None, "last_year_crimes": 0},
            "quality": {"history_points": 0, "stability": "insufficient"},
        }

    last_year = int(years[-1])
    last_total = int(totals[-1])
    target_year = max(int(target_year), last_year + 1)

    model_predictions = {}
    if has_sklearn and len(years) >= 4:
        X = build_features(years)
        X_target = build_features(np.array([target_year]))

        linear_model = LinearRegression()
        linear_model.fit(X[:, [0]], totals)
        model_predictions["Linear Regression"] = float(linear_model.predict(np.array([[target_year]]))[0])

        rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_model.fit(X, totals)
        model_predictions["Random Forest"] = float(rf_model.predict(X_target)[0])

        gb_model = GradientBoostingRegressor(random_state=42)
        gb_model.fit(X, totals)
        model_predictions["Gradient Boosting"] = float(gb_model.predict(X_target)[0])
    elif has_sklearn and len(years) >= 3:
        linear_model = LinearRegression()
        linear_model.fit(years.reshape(-1, 1), totals)
        model_predictions["Linear Regression"] = float(linear_model.predict(np.array([[target_year]]))[0])

    if not model_predictions:
        baseline = weighted_mean(totals[-min(3, len(totals)):])
        drift = weighted_mean(np.diff(totals[-min(4, len(totals)):])) if len(totals) > 1 else 0.0
        model_predictions["Weighted Momentum"] = float(baseline + drift)

    ensemble_value = float(np.mean(list(model_predictions.values())))
    predicted_value = int(max(0, round(ensemble_value)))

    diffs = np.diff(totals) if len(totals) > 1 else np.array([0.0])
    residual_span = max(float(np.std(diffs)) * 1.15, max(last_total * 0.06, 5000.0))
    confidence_low = int(max(0, round(predicted_value - residual_span)))
    confidence_high = int(max(confidence_low, round(predicted_value + residual_span)))
    delta_from_last_year = int(predicted_value - last_total)
    pct_change = float((delta_from_last_year / last_total) * 100) if last_total > 0 else 0.0

    if abs(pct_change) < 2:
        stability = "stable"
    elif abs(pct_change) < 7:
        stability = "moderate"
    else:
        stability = "volatile"

    return {
        "target_year": target_year,
        "predicted_crimes": predicted_value,
        "model_used": "Ensemble Forecast",
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "delta_from_last_year": delta_from_last_year,
        "pct_change_from_last_year": round(pct_change, 2),
        "ensemble_components": {k: int(max(0, round(v))) for k, v in model_predictions.items()},
        "historical_context": {
            "last_year": last_year,
            "last_year_crimes": last_total
        },
        "quality": {
            "history_points": int(len(years)),
            "stability": stability
        }
    }


def normalize_official_columns(df):
    rename_map = {
        "STATE": "state_name",
        "State": "state_name",
        "STATE_NAME": "state_name",
        "DISTRICT": "district_name",
        "District": "district_name",
        "DISTRICT_NAME": "district_name",
        "YEAR": "year",
        "Year": "year",
        "UT": "state_name",
        "Union Territory": "state_name",
    }
    df = df.rename(columns=rename_map).copy()
    df.columns = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    return df


def load_optional_official_2023(crime_cols):
    for candidate in OFFICIAL_2023_CANDIDATES:
        if not os.path.exists(candidate):
            continue
        try:
            if candidate.endswith(".csv"):
                df = pd.read_csv(candidate)
            else:
                df = pd.read_excel(candidate)
            df = normalize_official_columns(df)
            required = {"state_name", "district_name", "year"}
            if not required.issubset(df.columns):
                continue

            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"] == 2023].copy()
            if df.empty:
                continue

            for col in crime_cols:
                if col not in df.columns:
                    df[col] = 0

            keep_cols = ["state_name", "district_name", "year", *crime_cols]
            df = df[keep_cols].copy()
            df[crime_cols] = df[crime_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            return df, candidate
        except Exception:
            continue
    return None, None

def get_data():
    """Load and preprocess the dataset"""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
        
    data_path = os.path.join(PROJECT_ROOT, "districtwise-ipc-crimes.xlsx")
    if not os.path.exists(data_path):
        return None
        
    df = pd.read_excel(data_path, sheet_name="districtwise-ipc-crimes")
    cols_to_drop = ["id", "state_code", "district_code", "registration_circles"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    df["year"] = df["year"].astype(int)

    # Keep numeric aggregations limited to actual crime metrics, not admin/meta fields.
    excluded_numeric = {"year", "registration_circles"}
    crime_cols = [col for col in df.select_dtypes(include="number").columns if col not in excluded_numeric]
    df[crime_cols] = df[crime_cols].fillna(0)

    official_2023_df, _ = load_optional_official_2023(crime_cols)
    if official_2023_df is not None:
        df = df[df["year"] != 2023].copy()
        df = pd.concat([df, official_2023_df], ignore_index=True)

    df = df.copy()
    df["total_crimes"] = df[crime_cols].sum(axis=1)

    _DATA_CACHE = (df, crime_cols)
    warm_projection_cache_async(df, crime_cols, DEFAULT_PROJECTION_END_YEAR)
    return _DATA_CACHE

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "ml_enabled": has_sklearn})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Returns high-level statistics: total crimes, YoY growth, and top state.
    Can be filtered by state and year_range.
    """
    data = get_data()
    if not data:
        return jsonify({"error": "Dataset not found"}), 404
        
    df, crime_col_list = data
    
    # Optional Filtering
    state = request.args.get('state', 'all')
    try:
        min_year = int(request.args.get('min_year', df['year'].min()))
        max_year = int(request.args.get('max_year', df['year'].max()))
    except ValueError:
        return jsonify({"error": "Invalid year format"}), 400

    # Clamp year range to actual dataset bounds
    actual_min = int(df['year'].min())
    actual_max = int(df['year'].max())
    min_year = max(min_year, actual_min)
    max_year = min(max_year, actual_max)

    filtered_df = df[(df['year'] >= min_year) & (df['year'] <= max_year)]
    if state.lower() != 'all':
        filtered_df = filtered_df[filtered_df['state_name'].str.lower() == state.lower()]

    if filtered_df.empty:
         return jsonify({
             "total_crimes": 0,
             "yoy_growth": 0,
             "top_state": "None",
             "top_state_crimes": 0,
             "dominant_crime": "None",
             "dominant_crime_count": 0,
             "year_range": {"min": actual_min, "max": actual_max}
         })

    total_crimes = int(filtered_df['total_crimes'].sum())
    
    # Calculate YoY (Current Max Year vs Previous Year)
    yearly_totals = filtered_df.groupby('year')['total_crimes'].sum()
    yoy_growth = 0
    if len(yearly_totals) >= 2:
        last_year = yearly_totals.iloc[-1]
        prev_year = yearly_totals.iloc[-2]
        if prev_year > 0:
            yoy_growth = ((last_year - prev_year) / prev_year) * 100

    # Top State
    top_state = "Multiple"
    top_state_crimes = 0
    if state.lower() == 'all':
        state_totals = filtered_df.groupby('state_name')['total_crimes'].sum().sort_values(ascending=False)
        if not state_totals.empty:
            top_state = state_totals.index[0]
            top_state_crimes = int(state_totals.iloc[0])
    
    # Dominant Crime — FIX: explicitly sum each crime column and sort by value
    crime_sums = filtered_df[crime_col_list].sum()
    crime_sums = crime_sums.sort_values(ascending=False)  # Sort by actual total value
    dominant_crime = "None"
    dominant_crime_count = 0
    if not crime_sums.empty:
        dominant_crime = crime_sums.index[0].replace('_', ' ').title()
        dominant_crime_count = int(crime_sums.iloc[0])

    return jsonify({
        "total_crimes": total_crimes,
        "yoy_growth": round(yoy_growth, 2),
        "top_state": top_state,
        "top_state_crimes": top_state_crimes,
        "dominant_crime": dominant_crime,
        "dominant_crime_count": dominant_crime_count,
        "year_range": {"min": actual_min, "max": actual_max}
    })


@app.route('/api/chart/trend', methods=['GET'])
def get_trend_data():
    """Returns Yearly Total Crimes for charting, with a partial-year flag for the last year."""
    data = get_data()
    if not data:
         return jsonify({"error": "Dataset not found"}), 404
         
    df, _ = data
    state = request.args.get('state', 'all')
    
    if state.lower() != 'all':
        df = df[df['state_name'].str.lower() == state.lower()]
        
    yearly_totals = df.groupby('year')['total_crimes'].sum().reset_index()
    
    # Detect if the final year looks like partial data (< 60% of previous year)
    last_year_partial = False
    if len(yearly_totals) >= 2:
        last_total = yearly_totals['total_crimes'].iloc[-1]
        prev_total = yearly_totals['total_crimes'].iloc[-2]
        if prev_total > 0 and last_total < (prev_total * 0.6):
            last_year_partial = True

    labels = yearly_totals['year'].astype(str).tolist()
    values = yearly_totals['total_crimes'].tolist()
    
    return jsonify({
        "labels": labels,
        "values": values,
        "last_year_partial": last_year_partial,
        "last_year": int(yearly_totals['year'].iloc[-1]) if len(yearly_totals) > 0 else None
    })


@app.route('/api/predict', methods=['GET'])
def get_prediction():
    """
    ML Predictive Analysis Endpoint
    Forecasts a target year using historical data and linear regression.
    """
    data = get_data()
    if not data:
         return jsonify({"error": "Dataset not found"}), 404
         
    df, _ = data
    state = request.args.get('state', 'all')
    try:
        target_year = int(request.args.get('target_year', df['year'].max() + 1))
    except ValueError:
        return jsonify({"error": "Invalid target_year format"}), 400

    if state.lower() != 'all':
         df = df[df['state_name'].str.lower() == state.lower()]

    yearly_totals = df.groupby('year')['total_crimes'].sum().reset_index()
    years = yearly_totals["year"].values
    totals = yearly_totals["total_crimes"].values

    return jsonify(build_prediction_payload(years, totals, target_year))


@app.route('/api/districts', methods=['GET'])
def get_districts():
    """
    NEW: Returns distinct district names for a given state (from the real dataset).
    Query param: state (required)
    """
    data = get_data()
    if not data:
        return jsonify([]), 404

    df, _ = data
    state = request.args.get('state', '')

    if not state or state.lower() == 'all':
        # Return all distinct districts sorted
        districts = sorted(df['district_name'].dropna().unique().tolist())
    else:
        state_df = df[df['state_name'].str.lower() == state.lower()]
        districts = sorted(state_df['district_name'].dropna().unique().tolist())

    return jsonify(districts)


@app.route('/api/crime_types', methods=['GET'])
def get_crime_types():
    """
    NEW: Returns all actual crime type column names from the dataset (human-readable).
    """
    data = get_data()
    if not data:
        return jsonify([]), 404

    df, crime_cols = data
    # Exclude meta columns
    exclude = {'state_name', 'district_name', 'year', 'total_crimes'}
    crime_types = [
        {"key": col, "label": col.replace('_', ' ').title()}
        for col in crime_cols
        if col not in exclude
    ]
    # Sort by label
    crime_types.sort(key=lambda x: x['label'])
    return jsonify(crime_types)


@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """
    Returns the complete preprocessed dataset so the frontend 
    can maintain all its interactive charts, maps, and tables 
    without relying on fake mock generation.
    """
    data = get_data()
    if not data:
         return jsonify([]), 404
         
    df, _ = data
    
    export_df = df.copy()
    export_df["is_projected"] = False
    export_df["data_stage"] = "actual"
    
    rename_map = {
        'state_name': 'State',
        'district_name': 'District',
        'year': 'Year',
        'total_crimes': 'Total',
        'is_projected': 'IsProjected',
        'data_stage': 'DataStage'
    }
    
    export_df.rename(columns=rename_map, inplace=True)
    export_df.fillna(0, inplace=True)
    
    records = export_df.to_dict(orient='records')
    return jsonify(records)


@app.route('/api/projections_dataset', methods=['GET'])
def get_projections_dataset():
    """
    Returns modeled district-level forecast rows through a requested end year.
    """
    data = get_data()
    if not data:
         return jsonify([]), 404

    df, crime_cols = data
    latest_year = int(df["year"].max())
    try:
        end_year = int(request.args.get("end_year", DEFAULT_PROJECTION_END_YEAR))
    except ValueError:
        return jsonify({"error": "Invalid end_year format"}), 400

    end_year = max(latest_year + 1, min(end_year, latest_year + 3))
    records = build_projection_records(df, crime_cols, end_year=end_year)
    if not records:
        return jsonify([])
    return jsonify(records)


@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory(FRONTEND_ROOT, 'index.html')


@app.route('/<path:path>', methods=['GET'])
def serve_frontend(path):
    target_path = os.path.join(FRONTEND_ROOT, path)
    if os.path.isfile(target_path):
        return send_from_directory(FRONTEND_ROOT, path)
    return send_from_directory(FRONTEND_ROOT, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting CrimeScope API Backend on port {port}...")
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=port)
