import * as tf from "@tensorflow/tfjs-node";
import { generateTrainingData, tokenlize } from "../utils";

const config = {
  modelSavePath: "file://./src/model",
  xsDataSetPath: "./src/dataset/comments.txt",
  ysDataSetPath: "./src/dataset/rate.txt",
};

const train = async () => {
  tokenlize(config.xsDataSetPath);
  console.log("tokenlized");
  const { xs, ys } = generateTrainingData(
    config.xsDataSetPath,
    config.ysDataSetPath,
    20
  );

  console.log("training data generated");

  const input = tf.tensor2d(xs).reshape([xs.length, 20, 1]);
  const output = tf.oneHot(tf.tensor(ys).cast("int32"), 2);

  const model = tf.sequential();

  // todo
  model.add(
    tf.layers.lstm({
      inputShape: [20, 1],
      units: 256,
      returnSequences: true
    })
  );
  model.add(tf.layers.lstm({ units: 128, returnSequences: true }));
  model.add(tf.layers.lstm({ units: 64, }));
  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.1),
    metrics: ["accuracy"],
  });
  model.summary();
  await model.fit(input, output, {
    epochs: 10,
    shuffle: true,
    // @ts-ignore
    onEpochEnd: (epoch, logs) => {
      console.log("Epoch: " + epoch + " Loss: " + logs.loss);
    },
  });

  await model.save(config.modelSavePath);
};

export { train };
