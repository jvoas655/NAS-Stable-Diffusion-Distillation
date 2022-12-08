import requests
import os
import multiprocessing as mp
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
import pickle
import sys
from tqdm import tqdm


def grab(line):
    ROOT = "..\\data\\cc_data"
    """
    Download a single image from the TSV.
    """
    uid, split, line = line
    try:
        caption, url = line.split("\t")[:2]
    except:
        #print("Parse error")
        return

    if os.path.exists(ROOT+"/%s/%d/%d.jpg"%(split,uid%1000,uid)):
        #print("Finished", uid)
        return uid, caption, url

    # Let's not crash if anythign weird happens
    try:
        dat = requests.get(url, timeout=10)
        if dat.status_code != 200:
            #print("404 file", url)
            return

        # Try to parse this as an Image file, we'll fail out if not
        im = Image.open(BytesIO(dat.content))
        im.thumbnail((512, 512), PIL.Image.BICUBIC)
        if min(*im.size) < max(*im.size)/3:
            #print("Too small", url)
            return

        im.save(ROOT+"/%s/%d/%d.jpg"%(split,uid%1000,uid))

        # Another try/catch just because sometimes saving and re-loading
        # the image is different than loading it once.
        try:
            o = Image.open(ROOT+"/%s/%d/%d.jpg"%(split,uid%1000,uid))
            o = np.array(o)

            #print("Success", o.shape, uid, url)
            return uid, caption, url
        except:
            pass
            #print("Failed", uid, url)
            
    except Exception as e:
        #print("Unknown error", e)
        pass

if __name__ == "__main__":
    ROOT = "..\\data\\cc_data"

    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
        os.mkdir(os.path.join(ROOT,"train"))
        os.mkdir(os.path.join(ROOT,"val"))
        for i in range(1000):
            os.mkdir(os.path.join(ROOT,"train", str(i)))
            os.mkdir(os.path.join(ROOT,"val", str(i)))

    
    p = mp.Pool(60)
    
    for tsv in sys.argv[1:]:
        print("Processing file", tsv)
        assert 'val' in tsv.lower() or 'train' in tsv.lower()
        split = 'val' if 'val' in tsv.lower() else 'train'
        data = [(i,split,x) for i,x in enumerate(open(tsv, encoding='utf-8').read().split("\n"))]
        #data = list(filter(lambda d: d[0] < 200000, data))
        results = []
        with tqdm(total = len(data), desc=f"{split} - download") as tq:
            for result in p.imap(grab, data):
                results.append(result)
                tq.update(1)
        out = open(tsv.replace(".tsv","_output.csv"),"w", encoding='utf-8')
        out.write("title\tfilepath\n")
        
        for row in tqdm(results, total=len(results), desc = f"{split} - writing"):
            if row is None: continue
            id, caption, url = row
            fp = os.path.join(ROOT, split, str(id % 1000), str(id) + ".jpg")
            if os.path.exists(fp):
                out.write("%s\t%s\n"%(caption,fp))
            else:
                print("Drop", id)
        out.close()
        
    p.close()

