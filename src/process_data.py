
import os
from PIL import Image
import glob
from multiprocessing import Pool
from tqdm import tqdm

def process_img(args):
    src, out, size = args
    if (os.path.exists(out)):
        return
    if (not os.path.exists(src)):
        return
    try:
        with Image.open(src) as img:
            # resize the image to 512 x 512
            img = img.resize(size)
            
            # rotate the image if required
            # img = img.rotate(90)
            
            # save the resized image, modify the resample method if required, modify the output directory as well
            img.save(out, format="JPEG", resample=Image.Resampling.NEAREST)
    except:
        return

def process_folder(path, output_folder, size = (512, 512), threads = 32):
    args = []
    if (not os.path.exists(output_folder)):
        os.mkdir(output_folder)

    for root, dirs, files in os.walk(path):
        for file in files:

            args.append((root + "\\" + file, output_folder + file, size))
    with tqdm(total = len(args)) as tq:
        with Pool(threads) as pool:
            
            for result in pool.imap_unordered(process_img, args):
                tq.update(1)


if __name__ == "__main__":
    process_folder("..\\data\\cc_data\\train\\", "..\\data\\cc_data\\proc_train\\")
    process_folder("..\\data\\cc_data\\val\\", "..\\data\\cc_data\\proc_val\\")