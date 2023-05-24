import argparse
import sys
import json
import logging

from src import preprocessing, train, generate_by_names, generate_with_letter

logging.basicConfig(filename='cache/runtime.log',level=logging.DEBUG, filemode='w')
logging.debug("Starting...")

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", '-d', type=str, required=False,
                    help="A txt file containing data of names")

parser.add_argument("--config_file_path", '-c', type=str, required=False,
                    help="A json file containing configuration.")

parser.add_argument("--weight_path", '-w', type=str, required=False,
                    help="A file containing saved weights of model for generation.")

parser.add_argument("--mode", '-m', type=str, required=False,
                    help='Choose train/generate mode.\nMode "1": Training\nMode "2": Generate with a starting letter\nMode "3": Generate random names given a prompt letter.')

parser.add_argument("--prompt", '-p', type=str, required=False,
                    help="Input letter for generation")

parser.add_argument("--len", '-l', type=int, required=False,
                    help="Number of names to generate")

parser.add_argument("--output_path", '-out', type=str, required=False,
                    help="Path to name output files")

args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

if __name__ == "__main__":
    with open(args.config_file_path) as f:
        config = json.load(f)

    if args.mode == "1":
        logging.debug(f"Mode training, input path {args.data_path}")
        logging.debug("Preprocessing...")
        loader = preprocessing(args.data_path, **config)
        logging.debug("Training...")
        train(loader, **config)
        logging.debug("Training finished.")
    elif args.mode == "2":
        logging.debug(f"Mode generate with letter.\nWeight path: {args.weight_path}\nPrompt letter: {args.prompt}.\nLength: {args.len}")
        output = generate_with_letter(args.weight_path, args.prompt, args.len, **config)
        with open(args.output_path, "w") as file:
            file.write(output)
        logging.debug("Generating finished.")
    elif args.mode == "3":
        logging.debug(f"Mode generate random names.\nWeight path: {args.weight_path}\nPrompt letter: {args.prompt}.\nLength: {args.len}")
        output = generate_by_names(args.weight_path, args.prompt, args.len, **config)
        with open(args.output_path, "w") as file:
            file.write(output)
        logging.debug("Generating finished.")
