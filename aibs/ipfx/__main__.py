from ._schemas import InputParameters, OutputParameters
from argschema import ArgSchemaParser

def main():
    """Main entry point for running ipfx."""
    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)
    output = {}
    # YOUR STUFF GOES HERE
    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output)
    else:
        print(mod.get_output_json(output))

if __name__ == "__main__":
    main()
