import { Express, Request, Response, Router } from "express";
import { commonRes } from "../utils/response";
import { emotionPredict, emotionTrain } from "../lstm/emotion";
import { generateTextResponse, generationTrain } from "../lstm/generation";

interface RouterConf {
  path: string;
  router: Router;
  meta?: unknown;
}

// 路由配置
const routerConf: Array<RouterConf> = [
  {
    path: "/api",
    router: Router().get("/alive", async (req: Request, res: Response) => {
      const result = {
        status: "alive",
      };
      res.status(200).send(commonRes(result));
    }),
  },
  {
    path: "/api",
    router: Router().get("/emotion/train", async (req: Request, res: Response) => {
      emotionTrain();
      const result = {
        status: "training",
      };
      res.status(200).send(commonRes(result));
    }),
  },
  {
    path: "/api",
    router: Router().get("/generation/train", async (req: Request, res: Response) => {
      generationTrain();
      const result = {
        status: "training",
      };
      res.status(200).send(commonRes(result));
    }),
  },
  {
    path: "/api",
    router: Router().get("/emotion/predict", async (req: Request, res: Response) => {
      const result = {
        output: await emotionPredict(req.query.input as string),
      };
      res.status(200).send(commonRes(result));
    }),
  },
  {
    path: "/api",
    router: Router().get("/generation/predict", async (req: Request, res: Response) => {
      const result = {
        output: await generateTextResponse(req.query.input as string),
      };
      res.status(200).send(commonRes(result));
    }),
  },
];

function routes(app: Express) {
  // 根目录
  app.get("/", (req: Request, res: Response) =>
    res.status(200).send("express server")
  );

  routerConf.forEach((conf) => app.use(conf.path, conf.router));
}

export default routes;
