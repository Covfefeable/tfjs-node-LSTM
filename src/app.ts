import express from 'express'
import routes from './routes/index' // 路由
import logger from './utils/logger'
import 'dotenv/config'
import { ServerPort } from './utils/const'

const app = express()
app.use(express.json())

// 启动
app.listen(ServerPort, async () => {
  logger.info(`App is running at http://localhost:${ServerPort}`)
  routes(app)
})



