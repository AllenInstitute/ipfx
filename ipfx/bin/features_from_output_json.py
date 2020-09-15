import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import allensdk.core.json_utilities as ju
import logging

def main():
    parser = argparse.ArgumentParser(
            description="Process feature json files from pipeline output into a flat csv."
        )
    parser.add_argument('files', type=str, nargs='+', help='feature json file(s) to process')
    parser.add_argument('--output', default='features.csv', help='path to write output csv')
    parser.add_argument('--qc', action='store_true', help='include qc and fx failure info in csv')
    args = parser.parse_args()
    process_file_list(args.files, output=args.output, save_qc_info=args.qc)

def process_file_list(files, cell_ids=None, output=None, save_qc_info=False):
    index_var = "cell_name"
    records = []
    for i, file in enumerate(files):
        record = extract_pipeline_output(file, save_qc_info=save_qc_info)
        if cell_ids is not None:
            record[index_var] = cell_ids[i]
        else:
        # use the parent folder for an id
        # could be smarter and check specimen_id vs name
            record[index_var] = Path(file).parent.name
        records.append(record)
    ephys_df = pd.DataFrame.from_records(records, index=index_var)
    if output:
        ephys_df.to_csv(output)
    return ephys_df

# from cell_features.long_squares
ls_features = [
    "input_resistance",
    "tau",
    "v_baseline",
    "sag",
    "rheobase_i",
    "fi_fit_slope",
]
hero_sweep_features = [
    'adapt',
    'avg_rate',
    'latency',
    'first_isi',
    'mean_isi',
    "median_isi",
    "isi_cv",
]
rheo_sweep_features = [
    'latency',
    'avg_rate',
]
mean_sweep_features = [
    'adapt',
]
ss_spike_features = [
    'upstroke_downstroke_ratio',
    'threshold_v',
    'peak_v',
    'fast_trough_v',
]
ramp_spike_features = [
    'upstroke_downstroke_ratio',
    'threshold_v',
    'peak_v',
    'fast_trough_v',
    'trough_v',
    'threshold_i',
]
ls_spike_features = [
    'upstroke_downstroke_ratio',
    'threshold_v',
    'peak_v',
    # include all troughs?
    'fast_trough_v',
    'trough_v',
    # not in cell record
    'width',
    'upstroke',
    'downstroke',
]
spike_adapt_features = [
    'isi',
    'width',
    'upstroke',
    'downstroke',
    'threshold_v',
    'fast_trough_v',
]
invert_features = ["first_isi"]
spike_threshold_shift_features = ["trough_v", "fast_trough_v", "peak_v"]

chirp_features = [
    '3db_freq',
    'peak_freq',
    'peak_ratio',
]

def extract_pipeline_output(output_json, save_qc_info=False):
    output = ju.read(output_json)
    record = {}

    fx_output = output.get('feature_extraction', {})
    if save_qc_info:
        qc_state = output.get('qc', {}).get('cell_state')
        if qc_state is not None:
            record['failed_qc'] = cell_state.get('failed_qc', False)
            record['fail_message_qc'] = '; '.join(cell_state.get('fail_tags'))

        fx_state = fx_output.get('cell_state')
        if fx_state is not None:
            record['failed_fx'] = cell_state.get('failed_fx', False)
            record['fail_message_fx'] = cell_state.get('fail_fx_message')

    cell_features = fx_output.get('cell_features', {})
    if cell_features is not None:
        record.update(extract_fx_output(cell_features))
    return record

def extract_fx_output(cell_features):
    record = {}

    ramps = cell_features.get('ramps')
    if ramps is not None:
        mean_spike_0 = ramps["mean_spike_0"]
        add_features_to_record(ramp_spike_features, mean_spike_0, record, suffix="_ramp")

    short_squares = cell_features.get('short_squares')
    if short_squares is not None:
        mean_spike_0 = short_squares["mean_spike_0"]
        add_features_to_record(ss_spike_features, mean_spike_0, record, suffix="_short_square")

    chirps = cell_features.get('chirps')
    if chirps is not None:
        add_features_to_record(chirp_features, chirps, record, suffix="_chirp")

    offset_feature_values(spike_threshold_shift_features, record, "threshold_v")
    invert_feature_values(invert_features, record)

    long_squares_analysis = cell_features.get('long_squares')
    if long_squares_analysis is not None:
        record.update(get_complete_long_square_features(long_squares_analysis))
    
    return record

def get_complete_long_square_features(long_squares_analysis):
    record = {}
    add_features_to_record(ls_features, long_squares_analysis, record)

    sweep = long_squares_analysis.get('rheobase_sweep',{})
    add_features_to_record(rheo_sweep_features, sweep, record, suffix='_rheo')
    add_features_to_record(ls_spike_features, sweep["spikes"][0], record, suffix="_rheo")

    sweep = long_squares_analysis.get('hero_sweep',{})
    add_features_to_record(hero_sweep_features, sweep, record, suffix='_hero')
    add_features_to_record(ls_spike_features, sweep["spikes"][0], record, suffix="_hero")
    features = get_spike_adapt_ratio_features(spike_adapt_features, sweep["spikes"])
    add_features_to_record(features.keys(), features, record, suffix="_hero")

    sweeps = long_squares_analysis.get('spiking_sweeps',{})
    # TODO: work on dataframe / reuse code
    for feature in mean_sweep_features:
        key = feature+'_mean'
        feat_list = [sweep[feature] for sweep in sweeps if feature in sweep]
        record[key] = np.nanmean([x for x in feat_list if x is not None])
    
    offset_feature_values(spike_threshold_shift_features, record, "threshold_v")
    invert_feature_values(invert_features, record)
    return record

def offset_feature_values(features, record, relative_to):
    for feature in features:
        matches = [x for x in record if x.startswith(feature)]
        for match in matches:
            suffix = match[len(feature):]
            val = record.pop(match)
            feature_short = feature[:-2] #drop the "_v"
            record[feature_short + "_deltav" + suffix] = (val - record[relative_to+suffix]) if val is not None else None

def invert_feature_values(features, record):
    for feature in features:
        matches = [x for x in record if x.startswith(feature)]
        for match in matches:
            suffix = match[len(feature):]
            val = record.pop(match)
            record[feature + "_inv" + suffix] = 1/val if val is not None else None

def add_features_to_record(features, feature_data, record, suffix=""):
    record.update({feature+suffix: feature_data.get(feature) for feature in features})

def get_spike_adapt_ratio_features(features, spikes_set, nth_spike=5):
    suffix = '_adapt_ratio'
    record = {}
    nspikes = len(spikes_set)
    if 'isi' in features:
        for i in range(nspikes-1):
            spikes_set[i]['isi'] = spikes_set[i+1]['peak_t'] - spikes_set[i]['peak_t']
    for feature in features:
        if nspikes <= nth_spike:
            value = None
        else:
            nth = spikes_set[nth_spike-1].get(feature)
            first = spikes_set[0].get(feature)
            value = nth/first if nth is not None else None
        record.update({feature+suffix: value})
    return record

    specimen_ids = get_specimen_ids(ids, input_file, project, include_failed_cells, cell_count_limit)
    compile_lims_results(specimen_ids).to_csv(output_file)

if __name__ == "__main__":
    main()

    