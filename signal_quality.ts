import * as fs from 'fs';
import { DataFrame } from 'dataframe-js';
import { mean, iqr } from 'simple-statistics';
import { butterworthBandpassFilter, filtfilt } from 'timeseries-analysis';
import { welchMethod } from 'fft-js';
import * as pickle from 'pickle';

class SignalQuality {
    private useEnsemble: boolean;
    private model1_3s: any;
    private model1_5s: any;
    private model2_3s: any;
    private model2_5s: any;
    private model3_3s: any;
    private model3_5s: any;
    private iir_b: number[];
    private iir_a: number[];
    private ecgFiltered: number[];
    private featureNames: string[];

    constructor(useEnsemble: boolean = false) {
        this.useEnsemble = useEnsemble;
        this.loadModels();

        const filterParams = {
            order: 5,
            cutoffLow: 0.35,
            cutoffHigh: 45,
            type: 'bandpass',
            design: 'butter'
        };

        [this.iir_b, this.iir_a] = butterworthBandpassFilter(filterParams.order, 500, filterParams.cutoffLow, filterParams.cutoffHigh);
    }

    private loadModels(): void {
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
    }

    private filterEcg(ecg: number[]): void {
        this.ecgFiltered = filtfilt(this.iir_b, this.iir_a, ecg);
    }

    processEcgWindow(ecg: number[], winLen: number): any {
        this.filterEcg(ecg);
        const distFeatures = this.extractDistributionFeatures();
        const hfEnergy = this.extractFrequencyFeatures();
        const features = [...distFeatures, hfEnergy];
        const featuresDf = new DataFrame([features], this.featureNames);

        if (this.useEnsemble) {
            return this.ensemblePredict(featuresDf);
        } else {
            if (winLen === 3) {
                return this.model1_3s.predict(featuresDf);
            } else {
                return this.model1_5s.predict(featuresDf);
            }
        }
    }

    private extractDistributionFeatures(): number[] {
        const data = this.ecgFiltered.map(x => (x - mean(this.ecgFiltered)) / SignalQuality.stdDev(this.ecgFiltered));
        const distFeatures = [
            SignalQuality.stdDev(data),
            SignalQuality.skewness(data),
            SignalQuality.kurtosis(data),
            Math.max(...data) - Math.min(...data),
            iqr(data),
            SignalQuality.entropy(data),
            data.sort()[Math.floor(data.length * 0.95)]
        ];
        return distFeatures;
    }

    private extractFrequencyFeatures(): number {
        const signal = this.ecgFiltered.map((val, idx, arr) => (idx > 0 ? Math.pow(val - arr[idx - 1], 2) : 0));
        const { spectrum } = welchMethod(signal, 500);
        const f100Index = spectrum.findIndex((val) => val.frequency >= 100);
        const hfEnergy = spectrum.slice(f100Index).reduce((acc, val) => acc + val.amplitude, 0);
        return hfEnergy;
    }

    private ensemblePredict(featuresDf: DataFrame): number {
        const model1Result = this.model1_3s.predict(featuresDf);
        const model2Result = this.model2_3s.predict(featuresDf);
        const model3Result = this.model3_3s.predict(featuresDf);
        const majorityVote = (model1Result + model2Result + model3Result >= 2) ? 1 : 0;
        return majorityVote;
    }

    static mean(data: number[]): number {
        return data.reduce((acc, x) => acc + x, 0) / data.length;
    }

    static stdDev(data: number[]): number {
        const mean = SignalQuality.mean(data);
        return Math.sqrt(data.reduce((acc, x) => acc + Math.pow(x - mean, 2), 0) / data.length);
    }

    static skewness(data: number[]): number {
        const n = data.length;
        if (n < 3) {
            return NaN;
        }
    
        const mean = data.reduce((acc, val) => acc + val, 0) / n;
        const sumOfCubedDifferences = data.reduce((acc, val) => acc + Math.pow(val - mean, 3), 0);
        const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
        const stdDev = Math.sqrt(variance);
    
        const skew = sumOfCubedDifferences / (n * Math.pow(stdDev, 3));
    
        return skew;
    }

    static kurtosis(data: number[]): number {
        const n = data.length;
        if (n < 4) {
            return NaN;
        }
    
        const mean = data.reduce((acc, val) => acc + val, 0) / n;
        const sumOfFourthDifferences = data.reduce((acc, val) => acc + Math.pow(val - mean, 4), 0);
        const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
        const stdDev = Math.sqrt(variance);
    
        const kurt = sumOfFourthDifferences / (n * Math.pow(stdDev, 4)) - 3;
    
        return kurt;
    }

    static entropy(data: number[]): number {
        const n = data.length;
        if (n === 0) {
            return NaN;
        }
    
        const counts: { [key: number]: number } = {};
        data.forEach((value) => {
            counts[value] = (counts[value] || 0) + 1;
        });
    
        return Object.values(counts).reduce((acc, count) => {
            const probability = count / n;
            return acc - probability * Math.log2(probability);
        }, 0);
    }
}


