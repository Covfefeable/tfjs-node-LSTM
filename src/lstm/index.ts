import * as tf from "@tensorflow/tfjs-node";

const config = {
  modelSavePath: "file://./src/model",
  dataSetPath: "./src/dataset/comments.txt",
};

const train = async () => {
    const model = tf.sequential();
};

export { train };
