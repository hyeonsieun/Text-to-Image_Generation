# Text-to-Image_Generation
![T2 Image Model](./T2Imodel.PNG)

This project was set as the final project assignment for the 2023 2nd OUTTA AI Bootcamp, where I served as the overall leader. This bootcamp is operated by [OUTTA](https://outta.ai), a non-profit AI education organization where I hold the position of president.<br>
<br>
This project is designed to generate images based on text input. The goal of this project is to implement a simple T2I (Text-to-Image) Generation model using Conditional GANs.<br>
<br>
Along with the OUTTA members, I created this project, set it as the final team project assignment for the 2023 2nd OUTTA AI Bootcamp, and evaluated the submissions to select the top-performing teams.<br>
<br>
If you're interested in undertaking this project yourself, you can download the skeleton code from [here](https://github.com/outta-ai/2023_OUTTA_AIBootcamp_final_project).<br>
<br>
This repository contains the solution for the project.<br>
<br>
For a more detailed explanation about this project, please refer to the uploaded '2023_final_project_guideline.pdf'.<br>
<br>
To execute this project, you'll need to modify the 'network.py' and 'train.py' files; it is recommended not to change other files.<br>
<br>
A brief explanatory video about this project is available at the following [link](https://www.youtube.com/watch?v=ZQsFbdTFZjo).<br>
<br>
This project was designed to be primarily executed in the Google Colab environment.
<br>
### MM-CelebA-HQ-Dataset

Dataset can be downloaded from [here](https://drive.google.com/drive/folders/1HwCTiyUUiN71fATB56Ea8qfUEq-X8AG7?usp=sharing).<br>
<br>
You can see the source of the dataset at the following [link](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset).<br>
<br>
Command for data preprocessing:

```
python preproc_datasets_celeba_zip_train.py --source=./multimodal_celeba_hq.zip \
                                            --dest train_data_6cap.zip --width 256 --height 256 \
                                            --transform center-crop --emb_dim 512 --width=256 --height=256
```

Zip file `./multimodal_celeba_hq.zip` is like:

```
./multimodal_celeba_hq.zip
  ├── image
  │   ├── 0.jpg
  │   ├── 1.jpg
  │   ├── 2.jpg
  │   └── ...
  └── celea-caption
  │   ├── 0.txt
  │   ├── 1.txt
  │   ├── 2.txt
  │   └── ...
``` 

---
## Measure FID and IS

If you want to measure FID and IS, run the file 'Evaluate_FID_and_IS.ipynb'.


## Reference 

This repository is implemented based on [LAFITE](https://github.com/drboog/Lafite), [StackGAN++](https://github.com/hanzhanggit/StackGAN-v2/tree/master) and [AttnGAN](https://github.com/taoxugit/AttnGAN/tree/master).
