# Predict Next Word from Scratch 🚀

Ever wondered how GPT predicts the next word? Let's build one from the ground up! This project is your guided tour through the evolution of language models - from the simplest bigram to the mighty transformer.

## What's Inside?

We'll journey through increasingly sophisticated models, implementing each one from scratch:

- Bigram Model - A look-up table to predict the next word.
- MLP ([Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))
- RNN ([Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf))
- GRU ([Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259))
- Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762))

All models are trained on a vocabulary of English words using `tiny_torch` - our minimalist PyTorch implementation that helps understand the fundamentals.

## Why From Scratch?

Building these models from scratch isn't just an exercise - it's the best way to truly understand how modern language models work. Each component is implemented with clear, educational code that prioritizes learning over efficiency.

## Acknowledgement

This project is largely inspired by Andrej Karpathy's excellent [makemore](https://github.com/karpathy/makemore) project. I add more models and more backpropagation components to the project.
