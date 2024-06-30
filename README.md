# Sound2Clip
Sound2Clip is a machine learning research program aimed at creating custom video clips based on audio input.   
At its core, it uses the DragonflAI API.   
To install dragonflAI:
`pip install git+https://gitlab.com/lr-technologies2/dragonflai.git`

## ROADMAP
### Experiment setup
- [x] Extract frames and audio from a video
- [x] Fourier preprocessing on audio sample
- [x] Create DataLoader with Fourier -> frame 
- [x] Visualise network output
- [x] Create a clip from a frames+audio
- [x] Create experiment 

### Dummy experiment
- [x] Find a dummy video to create an example
- [x] Create a first neural network architecture
- [x] Get first results

### Gather dataset
- [x] Find few clips to first train and test (~5clips)
- [ ] Improve dataset with other clips of the same genre (~100clips)
- [ ] Include more diversity in the dataset

### Improve network architecture
- [x] Train on one clip (audio, frames) -> Generate frames with same audio
- [ ] Train on one clip -> Generate frames with different audio
- [ ] Implement Variational latent space to improve generation diversity (train/test on the same clip )
- [ ] Implement Variational latent space to improve generation diversity (train clip1, test clip2)
- [ ] Implement sequential architecture to improve homogeneity in the frame generation
- [ ] Implement adversarial network to improve generation diversity
- [ ] Try Transformers to be able to train on more data
