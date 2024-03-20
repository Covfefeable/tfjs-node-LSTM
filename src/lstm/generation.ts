import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import { sentence2Token, tokenlize } from "../utils";
import log from "../utils/logger";

const config = {
  dataSetDir: "./src/dataset",
  modelDir: "./src/model/generation",
  maxSentenceLen: 20,
  step: 3,
};

const generationTrain = async () => {
  console.log("preparing training data, it may take a while");
  tokenlize(`${config.dataSetDir}/comments.txt`);

  const wordMap = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/wordMap.json`, "utf-8")
  );
  const wordSet = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/wordSet.json`, "utf-8")
  );
  const listWords = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/wordArr.json`, "utf-8")
  );
  console.log("word length:", listWords.length);
  console.log("unique word length:", wordSet.length);

  const sentences: string[][] = [];
  const nextWords: string[] = [];
  for (
    let i = 0;
    i < listWords.length - config.maxSentenceLen;
    i += config.step
  ) {
    sentences.push(listWords.slice(i, i + config.maxSentenceLen));
    nextWords.push(listWords[i + config.maxSentenceLen]);
  }

  const input = tf
    .zeros([sentences.length, config.maxSentenceLen])
    .arraySync() as number[][];
  const output = tf
    .zeros([sentences.length, wordSet.length])
    .arraySync() as number[][];

  console.log("input shape:", input.length, input[0].length);
  console.log("output shape:", output.length, output[0].length);

  sentences.forEach((sentence: string[], index) => {
    sentence.forEach((word: string, wordIndex) => {
      wordMap[word] && (input[index][wordIndex] = wordMap[word]);
    });
    output[index][wordMap[nextWords[index]]] = 1;
  });

  let model = null as any;
  // 如果存在模型则加载模型继续训练，如果 unique word 不一致则会报错
  if (fs.existsSync(`file://${config.modelDir}/model.json`)) {
    model = await tf.loadLayersModel(`${config.modelDir}/model.json`);
    console.log("existed");
  } else {
    model = tf.sequential();
    model.add(
      tf.layers.embedding({
        inputDim: wordSet.length,
        outputDim: 128,
        inputLength: config.maxSentenceLen,
      })
    );

    model.add(
      tf.layers.bidirectional({
        layer: tf.layers.lstm({ units: 64, returnSequences: true }),
      })
    );

    model.add(
      tf.layers.lstm({
        units: 64,
        returnSequences: true,
        goBackwards: true,
      })
    );
    model.add(tf.layers.flatten());
    model.add(
      tf.layers.dense({ units: wordSet.length, activation: "softmax" })
    );
  }

  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.rmsprop(0.002),
  });
  model.summary();

  const xs = tf.tensor2d(input as number[][]);
  const ys = tf.tensor2d(output as number[][]);

  await model.fit(xs, ys, {
    epochs: 20,
    batchSize: 32,
    callbacks: {
      onEpochEnd: async () => {
        await model.save(`file://${config.modelDir}`);
        const result = await generateTextResponse("你们");
        console.log(`result: ${result}`);
      },
    },
  });

  await model.save(`file://${config.modelDir}`);
};

const predict = async (input: string) => {
  const model = await tf.loadLayersModel(
    `file://${config.modelDir}/model.json`
  );
  const tokenizedInput = sentence2Token(input, config.maxSentenceLen);
  const inputTensor = tf.tensor2d([tokenizedInput]);
  const result = model.predict(inputTensor);
  const maxIndex = (result as tf.Tensor).argMax(1).dataSync()[0];
  const wordReverseMap = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/wordMapReverse.json`, "utf-8")
  );

  return wordReverseMap[maxIndex];
};

const generateTextResponse = async (input: string) => {
  let text = input;
  for (let i = 0; i < 20; i++) {
    text += await predict(text);
  }
  return text;
};

export { generationTrain, generateTextResponse };
