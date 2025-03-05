import pandas as pd
import numpy as np
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv("train.csv")
displayNum = df.loc[6]

pixels = displayNum[1:].values.reshape(28,28).astype(np.uint8)

img = Image.fromarray(pixels, mode='L')
img = img.resize((560, 560), resample=Image.NEAREST)
img.save(os.path.join(script_dir, "7img.png"))