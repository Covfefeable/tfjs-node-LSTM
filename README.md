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

* if running on windows, @tensorflow/tfjs-node version better be 3.18.0

* when you are trying to train the text generater model with onehot encoded input, note that the training data shouldn't be too large, otherwise nodejs will crash due to memory overflow. 2000 lines is a good choice to start with.

* if encountering memory overflow problem, try install the `increase-memory-limit` package globally and run the command `increase-memory-limit` from the root directory of the project.
