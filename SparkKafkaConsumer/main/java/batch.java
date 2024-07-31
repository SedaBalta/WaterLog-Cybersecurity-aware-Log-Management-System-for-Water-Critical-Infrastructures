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

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;


public class batch {

    private static final String MODEL_PATH="/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/modelnew";

    public static void main(String[] args) throws IOException {
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
                .appName("batch")
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


        // assuming df_raw is your DataFrame
        Dataset<Row> encodedDF = df
                .withColumn("Treat_Pump3Index", when(col("Treat_Pump3").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Treat_Pump2Index", when(col("Treat_Pump2").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Treat_Pump1Index", when(col("Treat_Pump1").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Treat_LimitSwitch1Index", when(col("Treat_LimitSwitch1").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump3Index", when(col("Dam_Pump3").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump2Index", when(col("Dam_Pump2").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump1Index", when(col("Dam_Pump1").equalTo(true), 1.0).otherwise(0.0))
                .withColumn("Dam_LimitSwitchIndex", when(col("Dam_LimitSwitch").equalTo(true), 1.0).otherwise(0.0));

        System.out.println("encodedDF");
        encodedDF.show();

        // Assuming encodedDF is your DataFrame
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Treat_Level2", "Treat_Level1", "Treat_CurrentVolume", "Dam_Level", "Dam_CurrentVolume",
                        "Treat_PumpFlowLiterMin", "Dam_Chlor_Raw", "Treat_Pump3Index", "Treat_Pump2Index", "Treat_Pump1Index",
                        "Treat_LimitSwitch1Index", "Dam_Pump3Index", "Dam_Pump2Index", "Dam_Pump1Index", "Dam_LimitSwitchIndex"})
                .setOutputCol("features");

        // Assuming encodedDF is your DataFrame
        Dataset<Row> vectorAssemblerDF = assembler.transform(encodedDF);
        System.out.println("vectorAssemblerDF");
        vectorAssemblerDF.show();
        vectorAssemblerDF.printSchema();

        LogisticRegression logisticregresyon=new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.00001)
                .setElasticNetParam(0.1)
                .setThreshold(0.1)
                .setFamily("multinomial");
        //multi kullanabilmek için .setfamily("multinominal") yapılmalı.https://spark.apache.org/docs/3.0.1/ml-classification-regression.html

        // Pipeline işlemleri başladı.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, logisticregresyon});
        PipelineModel model = pipeline.fit(encodedDF);

        // Modeli kaydet
        model.write().overwrite().save(MODEL_PATH);
        System.out.println("Saved");

        Dataset<Row> resultDF = model.transform(encodedDF);
        // Pipeline işlemleri tamamlandı.

        // Sonuçları göster
        System.out.println("resultDF:");
        resultDF.show();
        resultDF.printSchema();

        // Performans değerlendirmesi başladı.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(resultDF);
        System.out.println("Accuracy: " + accuracy);

        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = f1Evaluator.evaluate(resultDF);
        System.out.println("F1 Score: " + f1Score);

        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");
        double precision = precisionEvaluator.evaluate(resultDF);
        System.out.println("Precision: " + precision);

        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");
        double recall = recallEvaluator.evaluate(resultDF);
        System.out.println("Recall: " + recall);
// Performans değerlendirmesi tamamlandı.
    }
}
