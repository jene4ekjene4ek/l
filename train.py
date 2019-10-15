"""
PySpark Decision Tree Regression Example.
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow
import mlflow.spark
# from common import *



print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

metrics = ["rmse","r2", "mae"]

def train(maxDepth, maxBins):
    data = "0 128:51 129:159 130:253 131:159 132:50 155:48 156:238 157:252 158:252 159:252 160:237 182:54 183:227 184:253 185:252 186:239 187:233 188:252 189:57 190:6 208:10 209:60 210:224 211:252 212:253 213:252 214:202 215:84 216:252 217:253 218:122 236:163 237:252 238:252 239:252 240:253 241:252 242:252 243:96 244:189 245:253 246:167 263:51 264:238 265:253 266:253 267:190 268:114 269:253 270:228 271:47 272:79 273:255 274:168 290:48 291:238 292:252 293:252 294:179 295:12 296:75 297:121 298:21 301:253 302:243 303:50 317:38 318:165 319:253 320:233 321:208 322:84 329:253 330:252 331:165 344:7 345:178 346:252 347:240 348:71 349:19 350:28 357:253 358:252 359:195 372:57 373:252 374:252 375:63 385:253 386:252 387:195 400:198 401:253 402:190 413:255 414:253 415:196 427:76 428:246 429:252 430:112 441:253 442:252 443:148 455:85 456:252 457:230 458:25 467:7 468:135 469:253 470:186 471:12 483:85 484:252 485:223 494:7 495:131 496:252 497:225 498:71 511:85 512:252 513:145 521:48 522:165 523:252 524:173 539:86 540:253 541:225 548:114 549:238 550:253 551:162 567:85 568:252 569:249 570:146 571:48 572:29 573:85 574:178 575:225 576:253 577:223 578:167 579:56 595:85 596:252 597:252 598:252 599:229 600:215 601:252 602:252 603:252 604:196 605:130 623:28 624:199 625:252 626:252 627:253 628:252 629:252 630:233 631:145 652:25 653:128 654:252 655:253 656:252 657:141 658:37"

    (trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)

    # MLflow - log parameters
    print("Parameters:")
    print("  maxDepth:",maxDepth)
    print("  maxBins:",maxBins)
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)

    # Create pipeline
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, maxDepth=maxDepth, maxBins=maxBins)
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model and predic
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    # MLflow - log metrics
    print("Metrics:")
    predictions = model.transform(testData)
    for metric in metrics:
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric)
        v = evaluator.evaluate(predictions)
        print("  {}: {}".format(metric,v))
        mlflow.log_metric(metric,v)

    # MLflow - log model
    mlflow.spark.log_model(model, "spark-model")
    #mlflow.mleap.log_model(spark_model=model, sample_input=testData, artifact_path="mleap-model")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("App").getOrCreate()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="pyspark", required=False)
#     parser.add_argument("--data_path", dest="data_path", help="data_path", required=True)
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=2, type=int)
    parser.add_argument("--max_bins", dest="max_bins", help="max_bins", default=32, type=int)
    parser.add_argument("--describe", dest="describe", help="Describe data", default=False, action='store_true')
    args = parser.parse_args()

    client = mlflow.tracking.MlflowClient()
    print("experiment_name:",args.experiment_name)
    mlflow.set_experiment(args.experiment_name)
    print("experiment_id:",client.get_experiment_by_name(args.experiment_name).experiment_id)

#     data = read_data(spark, args.data_path)
#     data = spark.read.format("libsvm").load(args.data_path)
#     if (args.describe):
#         print("==== Data")
#         data.describe().show()
    

    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:",run.info.run_uuid)
        print("  experiment_id:",run.info.experiment_id)
        train(args.max_depth,args.max_bins)
