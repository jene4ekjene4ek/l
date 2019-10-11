"""
PySpark Decision Tree Classification Example.
"""
from __future__ import print_function

import sys, os
from argparse import ArgumentParser
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import mlflow
from mlflow import version
from mlflow import spark as mlflow_spark

  #from livy import LivySession


#LIVY_URL = 'http://172.16.80.22:8998'

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())
experiment_name = "pyspark"
print("experiment_name:", experiment_name)
mlflow.set_experiment(experiment_name)



print("Parameters: max_depth: {}  max_bins: {}".format(max_depth,max_bins))
    #spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

    # livy_session=LivySession(LIVY_URL)

    # Load the data stored in LIBSVM format as a DataFrame.
    #data_path = "../data.txt"
    #data = spark.read.format("libsvm").load(data_path)
data = "0 128:51 129:159 130:253 131:159 132:50 155:48 156:238 157:252 158:252 159:252 160:237 182:54 183:227 184:253 185:252 186:239 187:233 188:252 189:57 190:6 208:10 209:60 210:224 211:252 212:253 213:252 214:202 215:84 216:252 217:253 218:122 236:163 237:252 238:252 239:252 240:253 241:252 242:252 243:96 244:189 245:253 246:167 263:51 264:238 265:253 266:253 267:190 268:114 269:253 270:228 271:47 272:79 273:255 274:168 290:48 291:238 292:252 293:252 294:179 295:12 296:75 297:121 298:21 301:253 302:243 303:50 317:38 318:165 319:253 320:233 321:208 322:84 329:253 330:252 331:165 344:7 345:178 346:252 347:240 348:71 349:19 350:28 357:253 358:252 359:195 372:57 373:252 374:252 375:63 385:253 386:252 387:195 400:198 401:253 402:190 413:255 414:253 415:196 427:76 428:246 429:252 430:112 441:253 442:252 443:148 455:85 456:252 457:230 458:25 467:7 468:135 469:253 470:186 471:12 483:85 484:252 485:223 494:7 495:131 496:252 497:225 498:71 511:85 512:252 513:145 521:48 522:165 523:252 524:173 539:86 540:253 541:225 548:114 549:238 550:253 551:162 567:85 568:252 569:249 570:146 571:48 572:29 573:85 574:178 575:225 576:253 577:223 578:167 579:56 595:85 596:252 597:252 598:252 599:229 600:215 601:252 602:252 603:252 604:196 605:130 623:28 624:199 625:252 626:252 627:253 628:252 629:252 630:233 631:145 652:25 653:128 654:252 655:253 656:252 657:141 658:37"

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets
(trainingData, testData) = data.randomSplit([0.7, 0.3])
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=2, type=int)
    parser.add_argument("--max_bins", dest="max_bins", help="max_bins", default=32, type=int)
    args = parser.parse_args()
    current_file = os.path.basename(__file__)
    print("MLflow Version:", version.VERSION)

    client = mlflow.tracking.MlflowClient()
    print("experiment_id:",client.get_experiment_by_name(experiment_name).experiment_id)

    with mlflow.start_run() as run:
        print("run_id:", run.info.run_uuid)
        print("experiment_id:", run.info.experiment_id)
    # Train a DecisionTree model.
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_bins", max_bins)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=max_depth, maxBins=max_bins)

    # Chain indexers and tree in a Pipeline.
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error.
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    test_error = 1.0 - accuracy
    print("Test Error = {} ".format(test_error))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("test_error", test_error)

    treeModel = model.stages[2]
    print(treeModel)

    mlflow_spark.log_model(model, "spark-model")
    #mlflow.mleap.log_model(model, testData, "mleap-model") # TODO: Bombs :(

    spark.stop()
