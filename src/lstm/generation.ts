import * as tf from "@tensorflow/tfjs-node";
import nj from "numjs";
import fs from "fs";
import { sentence2Token, tokenlize } from "../utils";

const config = {
  dataSetDir: "./src/dataset",
  modelDir: "./src/model/generation",
  maxSentenceLen: 20,
  step: 3,
};

const generationTrain = async () => {
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

  const input = nj.zeros([
    sentences.length,
    config.maxSentenceLen,
    wordSet.length,
  ]);
  const output = nj.zeros([sentences.length, wordSet.length]);

  console.log("input shape:", input.shape);
  console.log("output shape:", output.shape);

  sentences.forEach((sentence: string[], index) => {
    sentence.forEach((word: string, wordIndex) => {
      wordMap[word] && input.set(index, wordIndex, wordMap[word], 1);
    });
    output.set(index, wordMap[nextWords[index]], 1);
  });

  let model = null;
  // 如果存在模型则加载模型继续训练，如果 unique word 不一致则会报错
  if (fs.existsSync(`file://${config.modelDir}/model.json`)) {
    model = await tf.loadLayersModel(`${config.modelDir}/model.json`);
    console.log("existed");
  } else {
    model = tf.sequential();
    model.add(
      tf.layers.lstm({
        units: 128,
        returnSequences: true,
        inputShape: [config.maxSentenceLen, wordSet.length],
      })
    );
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(
      tf.layers.lstm({
        units: 128,
        returnSequences: false,
      })
    );
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(
      tf.layers.dense({ units: wordSet.length, activation: "softmax" })
    );
  }

  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.rmsprop(0.002),
  });
  model.summary();

  const xs = tf.tensor3d(input.tolist() as number[][][]);
  const ys = tf.tensor2d(output.tolist() as number[][]);

  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 32,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(logs?.loss);
      },
    },
  });

  await model.save(`file://${config.modelDir}`);
};

const predict = async (input: string) => {
  const model = await tf.loadLayersModel(`file://${config.modelDir}/model.json`);
  const tokenizedInput = sentence2Token(input, config.maxSentenceLen);
  const wordSet = JSON.parse(
    fs.readFileSync(`${config.dataSetDir}/wordSet.json`, "utf-8")
  );
  const inputTensor = tf.oneHot(
    tf.tensor2d([tokenizedInput]).cast("int32"),
    wordSet.length
  );
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
