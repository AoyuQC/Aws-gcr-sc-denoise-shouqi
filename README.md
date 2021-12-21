.
   
    Denoise Models for speech application

    ├── README.md

    ├── scripts : scripts for generating training data
        ├── DataprepareAudio.py
        └── noise_mix

    ├── segan_modify: denoise model in Sagemaker
        ├── Experiment.ipynb
        ├── cfg
        ├── data
        └── source

    └── segan_raw: denoise model from github
        ├── README.md
        ├── assets
        ├── bnorm.py
        ├── cfg
        ├── clean_wav.sh
        ├── data_loader.py
        ├── discriminator.py
        ├── generator.py
        ├── main.py
        ├── make_tfrecords.py
        ├── model.py
        ├── ops.py
        ├── prepare_data.sh
        ├── requirements.txt
        └── train_segan.sh

9 directories, 17 files
