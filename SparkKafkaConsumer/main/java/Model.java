//import org.apache.spark.internal.config.R;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tree.impl.RandomForest;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static org.apache.spark.sql.functions.col;

public class Model {

    private static final String MODEL_PATH="/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/model";

    public static void main(String[] args) throws Exception {


        StructType schema = new StructType()
                .add("Treat_ScenarioID", DataTypes.DoubleType)
                .add("Treat_Level2", DataTypes.DoubleType)
                .add("Treat_Level1", DataTypes.DoubleType)
                .add("Treat_CurrentVolume", DataTypes.DoubleType)
                .add("Dam_Level", DataTypes.DoubleType)
                .add("Dam_CurrentVolume", DataTypes.DoubleType)
                .add("Treat_PumpFlowLiterMin", DataTypes.DoubleType)
                .add("Treat_Pump3", DataTypes.StringType)
                .add("Treat_Pump2",DataTypes.StringType)
                .add("Treat_Pump1", DataTypes.StringType)
                .add("Treat_LimitSwitch1", DataTypes.StringType)
                .add("Dam_Pump3", DataTypes.StringType)
                .add("Dam_Pump2", DataTypes.StringType)
                .add("Dam_Pump1", DataTypes.StringType)
                .add("Dam_LimitSwitch", DataTypes.StringType)
                .add("Dam_Chlor_Raw", DataTypes.DoubleType)
                .add("label",DataTypes.DoubleType);

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaStructuredNetworkWordCount")
                .config("spark.master", "local")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> df=spark
                .read()
                .option("header",false)
                .format("csv")
                .schema(schema)
                .load("/home/waterquality/IdeaProjects/SparkKafkaConsumer/src/main/resources/training.csv");

        df.printSchema();
        df.show();

        df.createOrReplaceTempView("empdata");
        @SuppressWarnings("unchecked")
        Dataset<Row> selectedDF=spark.sql("select * from empdata");

        selectedDF.show();


        List<String> columnsToIndex = Arrays.asList("Treat_Pump3","Treat_Pump2","Treat_Pump1","Treat_LimitSwitch1","Dam_Pump3","Dam_Pump2","Dam_Pump1","Dam_LimitSwitch");

        for (String colName : columnsToIndex) {
            StringIndexerModel indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "_index")
                    .fit(df);


            df = indexer.transform(df).drop(colName);

                   // Indexlenmiş sütunu orijinal sütunun yerine yerleştirme

            df = df.withColumn(colName, col(colName + "_index")).drop(colName + "_index");
        }

        df.show();
        df.printSchema();


        VectorAssembler vectorAssembler=new VectorAssembler() //indextostring metodu ile
                .setInputCols(new String[]{"Treat_Level2","Treat_Level1","Treat_CurrentVolume","Dam_Level","Dam_CurrentVolume","Treat_PumpFlowLiterMin",
                        "Treat_Pump3","Treat_Pump2","Treat_Pump1","Treat_LimitSwitch1","Dam_Pump3","Dam_Pump2","Dam_Pump1","Dam_LimitSwitch"})
                .setOutputCol("features");

        Dataset<Row> output = vectorAssembler.transform(df);

        output.show();
        output.printSchema();

   //     NaiveBayes nb = new NaiveBayes();

        LogisticRegression logisticregresyon=new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.00001)
                .setElasticNetParam(0.1)
                .setThreshold(0.1)
                .setFamily("multinomial");
        //multi kullanabilmek için .setfamily("multinominal") yapılmalı.https://spark.apache.org/docs/3.0.1/ml-classification-regression.html

        Pipeline pipeline=new Pipeline()
                .setStages(new PipelineStage[]{vectorAssembler,logisticregresyon});


        Dataset<Row>[] splitDF=df.randomSplit(new double[] {0.8,0.2});
        Dataset<Row> trainingData = splitDF[0];
        Dataset<Row> testData = splitDF[1];

        PipelineModel pipelineModel=pipeline.fit(trainingData);
        Dataset<Row> predictionDF=pipelineModel.transform(testData);
        predictionDF.show(false);


      MulticlassClassificationEvaluator
               multiclassClassificationEvaluator1=new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");
        MulticlassClassificationEvaluator
                multiclassClassificationEvaluator2=new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("precisionByLabel");
        MulticlassClassificationEvaluator
                multiclassClassificationEvaluator3=new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("recallByLabel");
        MulticlassClassificationEvaluator
                multiclassClassificationEvaluator4=new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double accuracy = multiclassClassificationEvaluator1.evaluate(predictionDF);
        double precisionByLabel = multiclassClassificationEvaluator2.evaluate(predictionDF);
        double recallByLabel = multiclassClassificationEvaluator3.evaluate(predictionDF);
        double f1 = multiclassClassificationEvaluator4.evaluate(predictionDF);

        System.out.println("Test set accuracy = " + accuracy);
        System.out.println("Test set precisionByLabel = " + precisionByLabel);
        System.out.println("Test set recallByLabel = " + recallByLabel);
        System.out.println("Test set f1 = " + f1);

        pipelineModel.write().overwrite().save(MODEL_PATH);
        System.out.println("Model kaydedildi.");

    }
}