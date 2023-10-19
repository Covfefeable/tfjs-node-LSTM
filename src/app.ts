import express from 'express'
import routes from './routes/index' // 路由
import logger from './utils/logger'
import 'dotenv/config'

const app = express()
app.use(express.json())

// 启动
app.listen(process.env.PORT, async () => {
  logger.info(`App is running at http://localhost:${process.env.PORT}`)
  routes(app)
})