import base64
import os
from pathlib import Path
from io import BytesIO
import time
from PIL import Image

from dalle_model import DalleModel
import streamlit as st

from consts import DEFAULT_IMG_OUTPUT_DIR, ModelSize

model_version= ModelSize.MINI

def base64_to_image(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def generate_images(text_prompt):
    num_images = 1
    img_format = "jpeg"
    save_to_disk = False
    output_dir = DEFAULT_IMG_OUTPUT_DIR
    dalle_model = DalleModel(model_version)
    generated_imgs = dalle_model.generate_images(text_prompt, num_images)


    generated_images = []
    if save_to_disk: 
        dir_name = os.path.join(output_dir,f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{text_prompt}")
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(generated_imgs):
        if save_to_disk: 
          img.save(os.path.join(dir_name, f'{idx}.{img_format}'), format=img_format)

        buffered = BytesIO()
        img.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        generated_images.append(img_str)

    
    return generated_images


def st_ui():
    st.title('AI model drawing images from any text input')
    text = st.text_input("Describe image you want to see")
    button = st.button('Draw')
    if button:
        with st.spinner(text=f"Drawing {text}, it can take up to few minutes..."):
            generated_images = generate_images(text)
            
            for img in generated_images:
                image = base64_to_image(img)
                st.image(image, caption='Generated Image')
    

if __name__ == "__main__":
    st_ui()
