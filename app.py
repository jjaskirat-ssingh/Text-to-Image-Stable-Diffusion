# Imports 
from pyexpat import model
import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

# Functions
def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save("generatedimage.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

# Creating the app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Diffusion App")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 20), text_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()