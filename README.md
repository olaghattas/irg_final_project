# irg_final_project
Final project for the Information retrieval and generation class


## For video understanding: VideoLLaMA3
multimodal foundation models with frontier image and video understanding capacity.

https://github.com/DAMO-NLP-SG/VideoLLaMA3/tree/main

what I have tested:
this model DAMO-NLP-SG/VideoLLaMA3-7B and 	DAMO-NLP-SG/VideoLLaMA3-2B and 7b Gave better results.

I followed the instruction to install what is needed and created apython env to run things. 
test with this command usign gradio:
python inference/launch_gradio_demo.py --model-path DAMO-NLP-SG/VideoLLaMA3-7B

the page to open after running the scriot http://0.0.0.0:9999/
example of the outputs
<img width="451" height="392" alt="image" src="https://github.com/user-attachments/assets/0c7440df-614d-41b3-9897-b2c17fddf4c5" />

the video sent:
https://github.com/user-attachments/assets/b4103703-ee1c-46b6-81a5-d64f43c011bf



<img width="451" height="392" alt="image" src="https://github.com/user-attachments/assets/ce9dec87-e1ce-4128-aa08-8ba1f352d22e" />

the video sent:
https://github.com/user-attachments/assets/2c191065-f046-4223-9686-78b5cf255612

## For video segmentation: Deva
https://github.com/hkchengrex/Tracking-Anything-with-DEVA

TODO: add the txt instruction file.

I sent the first video above and got 

timestep 1
<img width="320" height="240" alt="000001" src="https://github.com/user-attachments/assets/e59227ed-d9f8-4314-be6c-d7295ddc2fcc" />

timestep 61
<img width="320" height="240" alt="000061" src="https://github.com/user-attachments/assets/c96f3969-d206-4a3c-b3f5-5eba7b64a33d" />

last step time 92
<img width="320" height="240" alt="000092" src="https://github.com/user-attachments/assets/466ca96a-068b-40e2-b3ca-bba98b23aa5b" />
