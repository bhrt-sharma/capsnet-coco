import requests
import zipfile
import argparse
import io


DOWNLOAD_LINKS = {
	2014: {
		"train": "http://images.cocodataset.org/zips/train2014.zip",
		"val": "http://images.cocodataset.org/zips/val2014.zip",
		"test": "http://images.cocodataset.org/zips/test2014.zip"
	},
	2017: {
		"train": "http://images.cocodataset.org/zips/train2017.zip",
		"val": "http://images.cocodataset.org/zips/val2017.zip",
		"test": "http://images.cocodataset.org/zips/test2017.zip"
	}
}


def download(args):
	if args.dataset == "all":
		datasets = ["train", "val", "test"]
	else:
		datasets = [args.dataset]

	for dataset in datasets:
		download_link = DOWNLOAD_LINKS[args.year][dataset]

		print("Downloading {} {} dataset ... ".format(args.year, dataset))
		r = requests.get(download_link, stream=True)
		with open('{}_{}.zip'.format(args.year, dataset), 'wb') as f:
		    for chunk in r.iter_content(chunk_size=1024): 
		        if chunk: # filter out keep-alive new chunks
		            f.write(chunk)

		print("Extracting... ")
		zip_ref = zipfile.ZipFile('{}_{}.zip'.format(args.year, dataset), 'r')

		if dataset == "train":
			folder = args.train_folder
		elif dataset == "test":
			folder = args.test_folder
		elif dataset == "val":
			folder = args.val_folder

		zip_ref.extractall(folder)
		zip_ref.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-ds", 
		"--dataset",
	    default="all",
	    help="which dataset to download of train, test, or val",
	    choices=["train", "test", "val"]
	)
	parser.add_argument(
		"-y", 
		"--year",
	    default=2014,
	    help="which year, 2014 or 2017",
	    choices=[2014, 2017]
	)
	parser.add_argument(
		"-trf", 
		"--train_folder",
	    default="data/train/images"
	)
	parser.add_argument(
		"-tef", 
		"--test_folder",
	    default="data/test/images"
	)
	parser.add_argument(
		"-vf", 
		"--val_folder",
	    default="data/val/images"
	)
	args = parser.parse_args()
	download(args)


if __name__ == '__main__':
	main()