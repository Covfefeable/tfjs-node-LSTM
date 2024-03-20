# tensorflow-nodejs-lstm
stock comment generator & emotion analysis based on LSTM

## Introduction

```bash
# if pnpm is not installed
npm i -g pnpm

# install dependencies
pnpm i

# launch the express server locally
pnpm run dev

# train the model by sending a get request, for more api details, see the ./src/routes/ folder
curl http://127.0.0.1:1337/api/generation/train

```

## Note

* when you are trying to train the text generater model with onehot encoded input, note that the training data shouldn't be too large, otherwise nodejs will crash due to memory overflow. 800 lines is a good choice to start with.
