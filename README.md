# 📊 CrimeScope - Professional Crime Intelligence Dashboard

![Landing Page](https://github.com/user-attachments/assets/4c525593-e32b-4383-81e0-9a9e22460d04)

## 🎯 Overview

**CrimeScope** is a professional, enterprise-level analytics platform for comprehensive crime analysis across India. Built with Streamlit and powered by advanced machine learning, it provides interactive visualizations, predictive analytics, and detailed insights into district-wise IPC crimes.

This platform transforms raw crime data into actionable intelligence through:
- **Multi-page professional dashboard** with intuitive navigation
- **Interactive KPI cards** with year-over-year comparisons
- **Geographic visualizations** including choropleth maps and treemaps
- **Advanced trend analysis** with growth rates and moving averages
- **Deep dive analytics** with correlation matrices and comparisons
- **ML-powered predictions** using ensemble models (Linear Regression, Random Forest, Gradient Boosting)

---

## ✨ Key Features

### 🏠 **Landing Page**
- Executive summary with 5 key KPI cards
- Visual trend overview charts
- Quick insights and state rankings
- Data coverage statistics
- Navigation guide to all sections

### 📊 **Dashboard** (Page 1)
- Comprehensive analytics overview
- Interactive filters (states, districts, years, crime types)
- Real-time KPI calculations
- Multiple visualization tabs:
  - Trends Analysis
  - District Rankings
  - Crime Distribution
  - Detailed Statistics
- CSV export functionality

### 🗺️ **Geographic Analysis** (Page 2)
- **Choropleth maps** showing crime distribution by state
- **Treemap** and **Sunburst** charts for hierarchical visualization
- **State-Crime Type Heatmaps**
- Top districts analysis
- Regional comparisons
- State-level and district-level data exports

### 📈 **Trends Analysis** (Page 3)
- Time series visualizations with trend lines
- **Year-over-Year (YoY) comparisons**
- **Growth rate analysis** including CAGR
- **Moving averages** for smoothing trends
- Peak and low annotations
- Distribution analysis with box plots
- Statistical summaries

### 🔍 **Deep Dive Analysis** (Page 4)
Four analysis modes:
1. **Crime Type Analysis**: Detailed breakdowns with rankings
2. **State Comparison**: Multi-state analytics with heatmaps
3. **District Comparison**: Top districts across states
4. **Correlation Analysis**: Crime type relationships and patterns

### 🤖 **Predictions & Forecasting** (Page 5)
- **Three ML models** trained and compared:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Ensemble predictions** for robust forecasts
- Model performance metrics (MAE, RMSE, R²)
- Future forecasts (1-10 years ahead)
- Individual crime type forecasts
- Confidence intervals and trend indicators

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/PraneshC-143/IPC-State-wise-Crime-Analysis-V2.git
cd IPC-State-wise-Crime-Analysis-V2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

4. **Access the dashboard**
Open your web browser and navigate to `http://localhost:8501`

---

## 📦 Dependencies

The dashboard uses the following key packages:
- **streamlit** >= 1.28.0 - Web application framework
- **pandas** >= 2.0.0 - Data manipulation
- **plotly** >= 5.14.0 - Interactive visualizations
- **scikit-learn** >= 1.3.0 - Machine learning models
- **numpy** >= 1.24.0 - Numerical computing
- **folium** >= 0.14.0 - Geographic maps
- **geopy** >= 2.3.0 - Geocoding

*See `requirements.txt` for complete list*

---

## 📊 Data Coverage

### Built-in Dataset
- **Years**: 2017-2022 (6 years)
- **Source**: District-wise IPC crimes Excel file
- **Records**: 5,322 district-level entries
- **States**: 36 states and union territories
- **Districts**: 749 districts
- **Crime Types**: 117 IPC crime categories

### Data Structure
Each record contains:
- State name
- District name
- Year
- 117 specific IPC crime categories (murder, theft, robbery, etc.)
- Aggregated totals

---

## 🎨 Dashboard Pages

### Navigation
The dashboard features a multi-page architecture accessible via the sidebar:

1. **🏠 streamlit app** - Landing page with executive summary
2. **📊 Dashboard** - Comprehensive analytics with filters
3. **🗺️ Geographic Analysis** - Maps and regional insights
4. **📈 Trends Analysis** - Time series and growth analysis
5. **🔍 Deep Dive** - Detailed breakdowns and comparisons
6. **🤖 Predictions** - ML-based forecasting

### Interactive Filters

All pages support powerful filtering:
- **States**: Multi-select dropdown
- **Districts**: Dependent on selected states
- **Year Range**: Slider with min/max years
- **Crime Types**: Select all or choose specific types
- **Quick Actions**: Reset filters button

---

## 🤖 Machine Learning & Predictions

### Models Used

The prediction system employs three complementary models:

1. **Linear Regression**
   - Best for: Linear trends
   - Advantages: Fast, interpretable
   - Use case: Baseline predictions

2. **Random Forest Regressor**
   - Best for: Non-linear patterns
   - Advantages: Robust to outliers
   - Use case: Complex relationships

3. **Gradient Boosting Regressor**
   - Best for: High accuracy
   - Advantages: Adaptive learning
   - Use case: Optimal performance

### Features Engineering
- Normalized year values
- Lag features (previous 1-2 years)
- Rolling mean trends (3-year window)
- Historical patterns

### Ensemble Approach
The final forecast combines all three models using weighted averaging for robust predictions with reduced variance.

### Performance Metrics
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R² Score** (0-1): Variance explained by model

---

## 📈 Key Performance Indicators (KPIs)

The dashboard calculates and displays:

1. **Total Crimes**: Absolute count with YoY change percentage
2. **Crime Rate**: Estimated per 100K population
3. **Most Affected State**: State with highest crime count
4. **Highest Crime Type**: Most prevalent crime category
5. **YoY Growth Rate**: Average annual growth rate
6. **Top 5 States**: Ranking by total crimes

Each KPI includes:
- 🟢 Green indicator for decreasing trends
- 🔴 Red indicator for increasing trends
- 🟡 Yellow indicator for stable trends

---

## 📥 Data Export

Export capabilities available across all pages:
- **CSV Export**: Filtered data, summaries, forecasts
- **Chart Export**: PNG format (via Plotly toolbar)
- **Summary Reports**: Key statistics and metrics
- **Yearly Aggregates**: Time series data
- **State/District Summaries**: Regional breakdowns

---

## 🎨 Design & Styling

### Color Scheme
- **Primary**: #1f77b4 (Blue)
- **Secondary**: #f0f2f6 (Light Gray)
- **Success**: Green shades
- **Warning**: Red/Orange shades
- **Background**: #ffffff (White)

### UI Features
- Responsive layout (wide mode)
- Professional card design with shadows
- Consistent typography
- Interactive tooltips and help icons
- Smooth animations and transitions

---

## 🛠️ Project Structure

```
IPC-State-wise-Crime-Analysis-V2/
├── streamlit_app.py              # Main landing page
├── pages/                         # Multi-page dashboard
│   ├── 1_📊_Dashboard.py         # Comprehensive analytics
│   ├── 2_🗺️_Geographic_Analysis.py  # Maps and regions
│   ├── 3_📈_Trends_Analysis.py   # Time series analysis
│   ├── 4_🔍_Deep_Dive.py         # Detailed breakdowns
│   └── 5_🤖_Predictions.py       # ML forecasting
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── kpi_calculator.py         # KPI calculations
│   ├── map_generator.py          # Geographic visualizations
│   └── export_utils.py           # Export functionality
├── data_loader.py                # Data loading and filtering
├── analytics.py                  # Statistical analysis
├── visualizations.py             # Chart generation
├── prediction.py                 # ML models and predictions
├── config.py                     # Configuration settings
├── .streamlit/
│   └── config.toml               # Streamlit theme config
├── requirements.txt              # Python dependencies
├── districtwise-ipc-crimes.xlsx  # Main dataset
└── README.md                     # Documentation
```

---

## 💡 Usage Guide

### Getting Started

1. **Launch the app** and explore the landing page
2. **Navigate** using the sidebar to different analysis pages
3. **Apply filters** in the sidebar to customize your view
4. **Interact** with charts by hovering, clicking, and zooming
5. **Export data** using the download buttons
6. **Generate predictions** on the Predictions page

### Tips

- 💡 **Hover over metrics** to see detailed explanations
- 💡 **Use multi-select** to compare multiple states
- 💡 **Try different time ranges** to spot trends
- 💡 **Explore all tabs** within each page for comprehensive insights
- 💡 **Download charts** using the Plotly camera icon
- 💡 **Reset filters** anytime using the Reset button

---

## 🔬 Advanced Features

### Correlation Analysis
- Discover relationships between crime types
- Identify positive and negative correlations
- Visual correlation matrices with color coding

### Seasonality Detection
- Identify patterns across years
- Spot recurring trends
- Analyze cyclical behaviors

### Comparative Analysis
- State vs State comparisons
- District rankings
- Crime type distributions
- Year-over-year changes

### Statistical Summaries
- Descriptive statistics (mean, median, std dev)
- Distribution analysis
- Outlier detection
- Confidence intervals

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 👥 Authors

- **Pranesh C** - Initial work and dashboard development
- Contributors - See GitHub contributors list

---

## 🙏 Acknowledgments

- **Indian Crime Data**: District-wise IPC crimes dataset
- **Streamlit**: For the excellent web framework
- **Plotly**: For interactive visualizations
- **scikit-learn**: For machine learning capabilities
- **Open Source Community**: For various libraries and tools

---

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation

---

## 🔄 Updates & Changelog

### Version 2.0 (Current)
- ✅ Multi-page professional dashboard architecture
- ✅ 5 specialized analysis pages
- ✅ Advanced KPI calculations with YoY comparisons
- ✅ Geographic visualizations (maps, treemaps, sunburst)
- ✅ Comprehensive trend analysis with growth rates
- ✅ Deep dive analytics with correlations
- ✅ Enhanced ML predictions with 3 ensemble models
- ✅ Professional styling and responsive design
- ✅ Data export capabilities
- ✅ Interactive filters across all pages

### Version 1.0
- Basic single-page dashboard
- Simple visualizations
- Basic predictions
- Limited filtering options

---

## 🎯 Future Enhancements

Planned features for future releases:
- [ ] Real-time data integration
- [ ] Advanced forecasting models (ARIMA, Prophet)
- [ ] User authentication and role-based access
- [ ] Custom report generation (PDF)
- [ ] Alert system for anomaly detection
- [ ] Mobile-responsive design improvements
- [ ] API endpoints for data access
- [ ] Integration with external crime databases

---

**Built with ❤️ using Streamlit | Powered by Python & Machine Learning**

*Last Updated: February 2026*
