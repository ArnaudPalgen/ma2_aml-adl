import pandas as pd
import argparse

def main(csv_file):
	selected_labels = [
		'drawing',
		'throwing ball',
		'making bed'
	]
	df = pd.read_csv(csv_file)
	df = df[df['label'].isin(selected_labels)]
	df.to_csv(csv_file, index=False)

if __name__ == '__main__':
	description = 'Helper script for selecting labels in kinetics videos.'
	p = argparse.ArgumentParser(description=description)
	p.add_argument(
		'csv_file',
		type=str,
		help=('CSV file containing the following format: '
			  'YouTube Identifier,Start time,End time,Class label'))
	main(**vars(p.parse_args()))