
import { Express, Request, Response, Router } from 'express'

interface RouterConf {
  path: string,
  router: Router,
  meta?: unknown
}

// 路由配置
const routerConf: Array<RouterConf> = []

function routes(app: Express) {
  // 根目录
  app.get('/', (req: Request, res: Response) => res.status(200).send('a simple express server'))

  routerConf.forEach((conf) => app.use(conf.path, conf.router))
}

export default routes
