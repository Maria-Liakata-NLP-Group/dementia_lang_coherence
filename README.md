

# This repository contains the implementation code for the paper titled [A Digital Language Coherence Marker for Monitoring Dementia](https://aclanthology.org/2023.emnlp-main.986.pdf)



## Table of Contents
 a
- [Data](#data)
- [Usage](#usage)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Data
This work utilizes the Pit dementia corpora. Access to the data is password-protected and restricted to individuals who have signed an agreement. For additional information, please visit the DementiaBank website.


## Usage



### Available models

####PLM
- 'BERT_large', 'BERT_base',  'RoBERTa', 'SBert', 
####Generative LM
- 'gpt2', 'gpt2-medium', 'gpt2-large', 'gptneo', 't5-base', 't5-large'
####Discriminative LM
-'BertCNN','CNN', 'coh_model'

### Run the models

1. PLM Tuning: `python train.py --model_name PLM XXX`
2. Discriminative Tuning: `python train_disc.py --model_name Discriminative LM XXX`
3. Generative Tuning: `python train_gen.py --model_name Generative LM XXX`


### Test the models

1. PLM Tuning: `python test.py --model_name PLM XXX`
2. Discriminative Tuning: `python test_disc.py --model_name Discriminative LM XXX`
3. Generative Tuning: `python test_gen.py --model_name Generative LM XXX`

## Contributing
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add your feature`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request



## Author
Dimitris Gkoumas. For more information, please visit [gkoumasd.github.io](https://gkoumasd.github.io)  


## License
If you find this project useful for your research, please consider citing it using the following BibTeX entry:


```bibtex
@inproceedings{gkoumas-etal-2023-digital,
  title = "A Digital Language Coherence Marker for Monitoring Dementia",
  author = "Gkoumas, Dimitris  and Tsakalidis, Adam  and Liakata, Maria",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  year = "2023",
  pages = "16021--16034"
}
