"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var dataframe_js_1 = require("dataframe-js");
var simple_statistics_1 = require("simple-statistics");
var timeseries_analysis_1 = require("timeseries-analysis");
var fft_js_1 = require("fft-js");
var pickle = require("pickle");
var SignalQuality = /** @class */ (function () {
    function SignalQuality(useEnsemble) {
        var _a;
        if (useEnsemble === void 0) { useEnsemble = false; }
        this.useEnsemble = useEnsemble;
        this.loadModels();
        var filterParams = {
            order: 5,
            cutoffLow: 0.35,
            cutoffHigh: 45,
            type: 'bandpass',
            design: 'butter'
        };
        _a = (0, timeseries_analysis_1.butterworthBandpassFilter)(filterParams.order, 500, filterParams.cutoffLow, filterParams.cutoffHigh), this.iir_b = _a[0], this.iir_a = _a[1];
    }
    SignalQuality.prototype.loadModels = function () {
        // Load models from files
        this.model1_3s = pickle.load(fs.readFileSync('model1_4_19_24.pkl'));
        this.model1_5s = pickle.load(fs.readFileSync('model5s_1_4_19_24.pkl'));
        if (this.useEnsemble) {
            this.model2_3s = pickle.load(fs.readFileSync('model3s_2_4_19_24.pkl'));
            this.model2_5s = pickle.load(fs.readFileSync('model3s_3_4_19_24.pkl'));
            this.model3_3s = pickle.load(fs.readFileSync('model5s_2_4_19_24.pkl'));
            this.model3_5s = pickle.load(fs.readFileSync('model5s_3_4_19_24.pkl'));
        }
        this.featureNames = ['std_dev max', 'skewness max', 'kurtosis min', 'range_of_data max', 'iqr_value max', 'data_entropy max', 'p95 max', 'hf_energy max'];
    };
    SignalQuality.prototype.filterEcg = function (ecg) {
        this.ecgFiltered = (0, timeseries_analysis_1.filtfilt)(this.iir_b, this.iir_a, ecg);
    };
    SignalQuality.prototype.processEcgWindow = function (ecg, winLen) {
        this.filterEcg(ecg);
        var distFeatures = this.extractDistributionFeatures();
        var hfEnergy = this.extractFrequencyFeatures();
        var features = __spreadArray(__spreadArray([], distFeatures, true), [hfEnergy], false);
        var featuresDf = new dataframe_js_1.DataFrame([features], this.featureNames);
        if (this.useEnsemble) {
            return this.ensemblePredict(featuresDf);
        }
        else {
            if (winLen === 3) {
                return this.model1_3s.predict(featuresDf);
            }
            else {
                return this.model1_5s.predict(featuresDf);
            }
        }
    };
    SignalQuality.prototype.extractDistributionFeatures = function () {
        var _this = this;
        var data = this.ecgFiltered.map(function (x) { return (x - (0, simple_statistics_1.mean)(_this.ecgFiltered)) / SignalQuality.stdDev(_this.ecgFiltered); });
        var distFeatures = [
            SignalQuality.stdDev(data),
            SignalQuality.skewness(data),
            SignalQuality.kurtosis(data),
            Math.max.apply(Math, data) - Math.min.apply(Math, data),
            (0, simple_statistics_1.iqr)(data),
            SignalQuality.entropy(data),
            data.sort()[Math.floor(data.length * 0.95)]
        ];
        return distFeatures;
    };
    SignalQuality.prototype.extractFrequencyFeatures = function () {
        var signal = this.ecgFiltered.map(function (val, idx, arr) { return (idx > 0 ? Math.pow(val - arr[idx - 1], 2) : 0); });
        var spectrum = (0, fft_js_1.welchMethod)(signal, 500).spectrum;
        var f100Index = spectrum.findIndex(function (val) { return val.frequency >= 100; });
        var hfEnergy = spectrum.slice(f100Index).reduce(function (acc, val) { return acc + val.amplitude; }, 0);
        return hfEnergy;
    };
    SignalQuality.prototype.ensemblePredict = function (featuresDf) {
        var model1Result = this.model1_3s.predict(featuresDf);
        var model2Result = this.model2_3s.predict(featuresDf);
        var model3Result = this.model3_3s.predict(featuresDf);
        var majorityVote = (model1Result + model2Result + model3Result >= 2) ? 1 : 0;
        return majorityVote;
    };
    SignalQuality.mean = function (data) {
        return data.reduce(function (acc, x) { return acc + x; }, 0) / data.length;
    };
    SignalQuality.stdDev = function (data) {
        var mean = SignalQuality.mean(data);
        return Math.sqrt(data.reduce(function (acc, x) { return acc + Math.pow(x - mean, 2); }, 0) / data.length);
    };
    SignalQuality.skewness = function (data) {
        var n = data.length;
        if (n < 3) {
            return NaN;
        }
        var mean = data.reduce(function (acc, val) { return acc + val; }, 0) / n;
        var sumOfCubedDifferences = data.reduce(function (acc, val) { return acc + Math.pow(val - mean, 3); }, 0);
        var variance = data.reduce(function (acc, val) { return acc + Math.pow(val - mean, 2); }, 0) / n;
        var stdDev = Math.sqrt(variance);
        var skew = sumOfCubedDifferences / (n * Math.pow(stdDev, 3));
        return skew;
    };
    SignalQuality.kurtosis = function (data) {
        var n = data.length;
        if (n < 4) {
            return NaN;
        }
        var mean = data.reduce(function (acc, val) { return acc + val; }, 0) / n;
        var sumOfFourthDifferences = data.reduce(function (acc, val) { return acc + Math.pow(val - mean, 4); }, 0);
        var variance = data.reduce(function (acc, val) { return acc + Math.pow(val - mean, 2); }, 0) / n;
        var stdDev = Math.sqrt(variance);
        var kurt = sumOfFourthDifferences / (n * Math.pow(stdDev, 4)) - 3;
        return kurt;
    };
    SignalQuality.entropy = function (data) {
        var n = data.length;
        if (n === 0) {
            return NaN;
        }
        var counts = {};
        data.forEach(function (value) {
            counts[value] = (counts[value] || 0) + 1;
        });
        return Object.values(counts).reduce(function (acc, count) {
            var probability = count / n;
            return acc - probability * Math.log2(probability);
        }, 0);
    };
    return SignalQuality;
}());
