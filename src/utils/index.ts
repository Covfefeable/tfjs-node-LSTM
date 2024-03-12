import fs from "fs";

const generateCharSetMap = (path: string): Promise<Array<string>> => {
  // 读取 path 文件内容，生成字符Map
  return new Promise((resolve, reject) => {
    fs.readFile(path, "utf-8", (err, data) => {
      if (err) {
        reject(err);
      } else {
        const charSet = new Set<string>();
        for (let i = 0; i < data.length; i++) {
          charSet.add(data[i]);
        }
        fs.writeFileSync(
          "./src/dataset/charSet.json",
          JSON.stringify(Array.from(charSet))
        );
        resolve(Array.from(charSet));
      }
    });
  });
};

export { generateCharSetMap };
