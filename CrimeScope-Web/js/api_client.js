/**
 * API Client for interacting with the Python Flask Backend
 */

const isLocalSplitDev =
    (window.location.protocol === 'file:') ||
    ((window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') &&
        window.location.port === '8000');

const API_URL = isLocalSplitDev
    ? 'http://127.0.0.1:5000/api'
    : `${window.location.origin}/api`;

const ApiClient = {
    /**
     * Fetch high-level statistics (Total Crimes, YoY Growth, Top State)
     */
    async getStats(state = 'all', minYear = 2017, maxYear = 2023) {
        try {
            const response = await fetch(`${API_URL}/stats?state=${encodeURIComponent(state)}&min_year=${minYear}&max_year=${maxYear}`);
            if (!response.ok) throw new Error('API Error fetching stats');
            return await response.json();
        } catch (error) {
            console.error(error);
            return null;
        }
    },

    /**
     * Fetch trend line chart data (Yearly totals) with partial-year detection
     */
    async getTrend(state = 'all') {
        try {
            const response = await fetch(`${API_URL}/chart/trend?state=${encodeURIComponent(state)}`);
            if (!response.ok) throw new Error('API Error fetching trend data');
            return await response.json();
        } catch (error) {
            console.error(error);
            return null;
        }
    },

    /**
     * Fetch the machine learning prediction for a target year
     */
    async getPrediction(state = 'all', targetYear = 2024) {
        try {
            const response = await fetch(`${API_URL}/predict?state=${encodeURIComponent(state)}&target_year=${targetYear}`);
            if (!response.ok) throw new Error('API Error fetching prediction');
            return await response.json();
        } catch (error) {
            console.error(error);
            return null;
        }
    },

    /**
     * Fetch the entire preprocessed dataset
     * Powers the interactive HTML UI (map, tables, and distribution charts)
     */
    async getDashboardData() {
        try {
            const response = await fetch(`${API_URL}/dataset`);
            if (!response.ok) throw new Error('API Error fetching dataset');
            return await response.json();
        } catch (error) {
            console.error(error);
            return [];
        }
    },

    /**
     * Fetch modeled forecast rows through a target end year.
     */
    async getProjectionDataset(endYear = 2025) {
        try {
            const response = await fetch(`${API_URL}/projections_dataset?end_year=${encodeURIComponent(endYear)}`);
            if (!response.ok) throw new Error('API Error fetching projection dataset');
            return await response.json();
        } catch (error) {
            console.error(error);
            return [];
        }
    },

    /**
     * NEW: Fetch distinct districts for a given state from the real dataset
     * @param {string} state - State name, or 'all' for all districts
     */
    async getDistricts(state = 'all') {
        try {
            const response = await fetch(`${API_URL}/districts?state=${encodeURIComponent(state)}`);
            if (!response.ok) throw new Error('API Error fetching districts');
            return await response.json();
        } catch (error) {
            console.error(error);
            return [];
        }
    },

    /**
     * NEW: Fetch all actual crime type column names from the dataset
     * Returns array of { key, label } objects
     */
    async getCrimeTypes() {
        try {
            const response = await fetch(`${API_URL}/crime_types`);
            if (!response.ok) throw new Error('API Error fetching crime types');
            return await response.json();
        } catch (error) {
            console.error(error);
            return [];
        }
    }
};

window.ApiClient = ApiClient;
