from ipfx.chirp import extract_chirp_features
import ipfx.offpipeline_utils as op
import argschema as ags

module = ags.ArgSchemaParser(schema_type=op.FeatureCollectionParameters)
# TODO: maybe argschema can take care of this logic with a custom field?
if module.args["input_file"]: 
    with open(module.args["input_file"], "r") as f:
        ids = [int(line.strip("\n")) for line in f]
    op.run_feature_collection(extract_chirp_features, ids=ids, **module.args)
else:
    op.run_feature_collection(extract_chirp_features, **module.args)