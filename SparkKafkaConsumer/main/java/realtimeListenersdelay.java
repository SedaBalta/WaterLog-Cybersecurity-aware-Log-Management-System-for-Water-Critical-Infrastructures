import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.VoidFunction2;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryListener;
import org.apache.spark.sql.streaming.StreamingQueryProgress;
import org.apache.spark.sql.streaming.Trigger;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static org.apache.spark.sql.functions.*;

public class realtimeListenersdelay {

    public static void myCustomFunc(Dataset<Row> df, long batchID) {
        // Prediction column = "0", topic name = "normalRealtime2"
        Dataset<Row> normalDF = df.filter(col("prediction").equalTo(0.0));
        Dataset<Row> normalDF2 = normalDF.withColumn("value",
                concat(
                        col("Treat_ScenarioID"), lit(","),
                        col("label"), lit(","),
                        col("prediction")
                ));

        normalDF2.select("value").write()
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("topic", "normalRealtime2")
                .save();

        // Prediction column = "1", topic name = "anormalRealtime2"
        Dataset<Row> anormalDF = df.filter(col("prediction").equalTo(1.0));
        Dataset<Row> anormalDF2 = anormalDF.withColumn("value",
                concat(
                        col("Treat_ScenarioID"), lit(","),
                        col("label"), lit(","),
                        col("prediction")
                ));

        anormalDF2.select("value").write()
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("topic", "anormalRealtime2")
                .save();
    }

    public static void main(String[] args) throws Exception {
        Logger.getLogger("org").setLevel(Level.ERROR);
        SparkSession spark = SparkSession.builder()
                .appName("RealtimeDeneme")
                .master("local[1]")
                .config("spark.driver.memory", "2g")
                .config("spark.executor.memory", "4g")
                .getOrCreate();

        String modelPath = "/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/modelnew";
        String checkPointDir = "/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/checkpoints/newcheckpoint12";

        StructType schema = new StructType()
                .add("Treat_ScenarioID", DataTypes.DoubleType)
                .add("Treat_Level2", DataTypes.DoubleType)
                .add("Treat_Level1", DataTypes.DoubleType)
                .add("Treat_CurrentVolume", DataTypes.DoubleType)
                .add("Dam_Level", DataTypes.DoubleType)
                .add("Dam_CurrentVolume", DataTypes.DoubleType)
                .add("Treat_PumpFlowLiterMin", DataTypes.DoubleType)
                .add("Treat_Pump3", DataTypes.StringType)
                .add("Treat_Pump2", DataTypes.StringType)
                .add("Treat_Pump1", DataTypes.StringType)
                .add("Treat_LimitSwitch1", DataTypes.StringType)
                .add("Dam_Pump3", DataTypes.StringType)
                .add("Dam_Pump2", DataTypes.StringType)
                .add("Dam_Pump1", DataTypes.StringType)
                .add("Dam_LimitSwitch", DataTypes.StringType)
                .add("Dam_Chlor_Raw", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);

        Dataset<Row> data = spark.readStream()
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "jsonproducerp3")
                .load()
                .selectExpr("CAST(value AS STRING)");

        Dataset<Row> jsonDataDF = data.select(from_json(col("value"), schema).as("data")).select("data.*");

        Dataset<Row> encodedDF = jsonDataDF
                .withColumn("Treat_Pump3Index", when(col("Treat_Pump3").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Treat_Pump2Index", when(col("Treat_Pump2").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Treat_Pump1Index", when(col("Treat_Pump1").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Treat_LimitSwitch1Index", when(col("Treat_LimitSwitch1").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump3Index", when(col("Dam_Pump3").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump2Index", when(col("Dam_Pump2").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_Pump1Index", when(col("Dam_Pump1").equalTo("true"), 1.0).otherwise(0.0))
                .withColumn("Dam_LimitSwitchIndex", when(col("Dam_LimitSwitch").equalTo("true"), 1.0).otherwise(0.0));

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Treat_Level2", "Treat_Level1", "Treat_CurrentVolume", "Dam_Level", "Dam_CurrentVolume", "Treat_PumpFlowLiterMin", "Dam_Chlor_Raw",
                        "Treat_Pump3Index", "Treat_Pump2Index", "Treat_Pump1Index", "Treat_LimitSwitch1Index", "Dam_Pump3Index", "Dam_Pump2Index", "Dam_Pump1Index",
                        "Dam_LimitSwitchIndex"})
                .setOutputCol("features")
                .setHandleInvalid("keep");

        Dataset<Row> vectorAssemblerDF = assembler.transform(encodedDF);

        PipelineModel pipeModel = PipelineModel.load(modelPath);
        LogisticRegressionModel streamModel = (LogisticRegressionModel) pipeModel.stages()[1];

        Dataset<Row> transformedDF = streamModel.transform(vectorAssemblerDF);
        transformedDF.printSchema();

        StreamingQuery query = transformedDF.writeStream()
                .foreachBatch((VoidFunction2<Dataset<Row>, Long>) realtimeListenersdelay::myCustomFunc)
                .option("checkpointLocation", checkPointDir)
                .option("failOnDataLoss", "false")
                .trigger(Trigger.ProcessingTime("5 seconds"))
                .start();

        // Add a listener to track scheduling delay and write to a file
        // Add a listener to track scheduling delay and write to a file
        Set<Long> processedBatches = new HashSet<>();
        spark.streams().addListener(new StreamingQueryListener() {
            @Override
            public void onQueryStarted(QueryStartedEvent queryStartedEvent) {
            }

            @Override
            public void onQueryProgress(QueryProgressEvent queryProgressEvent) {
                StreamingQueryProgress progress = queryProgressEvent.progress();
                long batchId = progress.batchId();
                long triggerExecution = progress.durationMs().getOrDefault("triggerExecution", -1L);

                if (!processedBatches.contains(batchId)) {
                    System.out.println("Trigger Execution: " + triggerExecution + " ms");

                    // Write scheduling delay to a file
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter("scheduling_delays/ex1/scheduling_delays_5batch_1000ms_125akis_new.txt", true))) {
                        writer.write("Batch ID: " + batchId + ", Trigger Execution: " + triggerExecution + " ms\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    processedBatches.add(batchId);
                }
            }

            @Override
            public void onQueryTerminated(QueryTerminatedEvent queryTerminatedEvent) {
            }
        });

        try {
            query.awaitTermination();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
