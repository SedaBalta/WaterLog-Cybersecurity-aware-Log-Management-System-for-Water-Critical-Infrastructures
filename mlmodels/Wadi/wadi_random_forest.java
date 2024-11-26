package mlModels.waterlog;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;

public class wadi_random_forest {
    private static final String MODEL_PATH="/home/waterquality/IdeaProjects/SparkKafkaConsumer/resources/ml-models/waterlog/decisiontree";
    public static void main(String[] args) throws IOException{


        StructType schema = new StructType()
                .add("1_AIT_002_PV", DataTypes.DoubleType)
                .add("1_FIT_001_PV", DataTypes.DoubleType)
                .add("1_LS_001_AL", DataTypes.DoubleType)
                .add("1_LS_002_AL", DataTypes.DoubleType)
                .add("1_MV_001_STATUS", DataTypes.DoubleType)
                .add("1_MV_002_STATUS", DataTypes.DoubleType)
                .add("1_MV_003_STATUS", DataTypes.DoubleType)
                .add("1_MV_004_STATUS", DataTypes.DoubleType)
                .add("1_P_001_STATUS", DataTypes.DoubleType)
                .add("1_P_002_STATUS", DataTypes.DoubleType)
                .add("1_P_003_STATUS", DataTypes.DoubleType)
                .add("1_P_004_STATUS", DataTypes.DoubleType)
                .add("1_P_005_STATUS", DataTypes.DoubleType)
                .add("1_P_006_STATUS", DataTypes.DoubleType)
                .add("2_DPIT_001_PV", DataTypes.DoubleType)
                .add("2_FIC_101_CO", DataTypes.DoubleType)
                .add("2_FIC_101_PV", DataTypes.DoubleType)
                .add("2_FIC_101_SP", DataTypes.DoubleType)
                .add("2_FIC_201_CO", DataTypes.DoubleType)
                .add("2_FIC_201_PV", DataTypes.DoubleType)
                .add("2_FIC_201_SP", DataTypes.DoubleType)
                .add("2_FIC_301_CO", DataTypes.DoubleType)
                .add("2_FIC_301_PV", DataTypes.DoubleType)
                .add("2_FIC_301_SP", DataTypes.DoubleType)
                .add("2_FIC_401_CO", DataTypes.DoubleType)
                .add("2_FIC_401_PV", DataTypes.DoubleType)
                .add("2_FIC_401_SP", DataTypes.DoubleType)
                .add("2_FIC_501_CO", DataTypes.DoubleType)
                .add("2_FIC_501_PV", DataTypes.DoubleType)
                .add("2_FIC_501_SP", DataTypes.DoubleType)
                .add("2_FIC_601_CO", DataTypes.DoubleType)
                .add("2_FIC_601_PV", DataTypes.DoubleType)
                .add("2_FIC_601_SP", DataTypes.DoubleType)
                .add("2_FIT_001_PV", DataTypes.DoubleType)
                .add("2_FIT_002_PV", DataTypes.DoubleType)
                .add("2_FIT_003_PV", DataTypes.DoubleType)
                .add("2_FQ_101_PV", DataTypes.DoubleType)
                .add("2_FQ_201_PV", DataTypes.DoubleType)
                .add("2_FQ_301_PV", DataTypes.DoubleType)
                .add("2_FQ_401_PV", DataTypes.DoubleType)
                .add("2_FQ_501_PV", DataTypes.DoubleType)
                .add("2_FQ_601_PV", DataTypes.DoubleType)
                .add("2_LS_001_AL", DataTypes.DoubleType)
                .add("2_LS_002_AL", DataTypes.DoubleType)
                .add("2_LS_101_AH", DataTypes.DoubleType)
                .add("2_LS_101_AL", DataTypes.DoubleType)
                .add("2_LS_201_AH", DataTypes.DoubleType)
                .add("2_LS_201_AL", DataTypes.DoubleType)
                .add("2_LS_301_AH", DataTypes.DoubleType)
                .add("2_LS_301_AL", DataTypes.DoubleType)
                .add("2_LS_401_AH", DataTypes.DoubleType)
                .add("2_LS_401_AL", DataTypes.DoubleType)
                .add("2_LS_501_AH", DataTypes.DoubleType)
                .add("2_LS_501_AL", DataTypes.DoubleType)
                .add("2_LS_601_AH", DataTypes.DoubleType)
                .add("2_LS_601_AL", DataTypes.DoubleType)
                .add("2_LT_001_PV", DataTypes.DoubleType)
                .add("2_LT_002_PV", DataTypes.DoubleType)
                .add("2_MCV_007_CO", DataTypes.DoubleType)
                .add("2_MCV_101_CO", DataTypes.DoubleType)
                .add("2_MCV_201_CO", DataTypes.DoubleType)
                .add("2_MCV_301_CO", DataTypes.DoubleType)
                .add("2_MCV_401_CO", DataTypes.DoubleType)
                .add("2_MCV_501_CO", DataTypes.DoubleType)
                .add("2_MCV_601_CO", DataTypes.DoubleType)
                .add("2_MV_001_STATUS", DataTypes.DoubleType)
                .add("2_MV_002_STATUS", DataTypes.DoubleType)
                .add("2_MV_003_STATUS", DataTypes.DoubleType)
                .add("2_MV_004_STATUS", DataTypes.DoubleType)
                .add("2_MV_005_STATUS", DataTypes.DoubleType)
                .add("2_MV_006_STATUS", DataTypes.DoubleType)
                .add("2_MV_009_STATUS", DataTypes.DoubleType)
                .add("2_MV_101_STATUS", DataTypes.DoubleType)
                .add("2_MV_201_STATUS", DataTypes.DoubleType)
                .add("2_MV_301_STATUS", DataTypes.DoubleType)
                .add("2_MV_401_STATUS", DataTypes.DoubleType)
                .add("2_MV_501_STATUS", DataTypes.DoubleType)
                .add("2_MV_601_STATUS", DataTypes.DoubleType)
                .add("2_P_001_STATUS", DataTypes.DoubleType)
                .add("2_P_002_STATUS", DataTypes.DoubleType)
                .add("2_P_003_SPEED", DataTypes.DoubleType)
                .add("2_P_003_STATUS", DataTypes.DoubleType)
                .add("2_P_004_SPEED", DataTypes.DoubleType)
                .add("2_P_004_STATUS", DataTypes.DoubleType)
                .add("2_PIC_003_CO", DataTypes.DoubleType)
                .add("2_PIC_003_PV", DataTypes.DoubleType)
                .add("2_PIC_003_SP", DataTypes.DoubleType)
                .add("2_PIT_002_PV", DataTypes.DoubleType)
                .add("2_PIT_003_PV", DataTypes.DoubleType)
                .add("2A_AIT_002_PV", DataTypes.DoubleType)
                .add("3_AIT_001_PV", DataTypes.DoubleType)
                .add("3_AIT_002_PV", DataTypes.DoubleType)
                .add("3_AIT_003_PV", DataTypes.DoubleType)
                .add("3_AIT_004_PV", DataTypes.DoubleType)
                .add("3_FIT_001_PV", DataTypes.DoubleType)
                .add("3_LS_001_AL", DataTypes.DoubleType)
                .add("3_LT_001_PV", DataTypes.DoubleType)
                .add("3_MV_001_STATUS", DataTypes.DoubleType)
                .add("3_MV_002_STATUS", DataTypes.DoubleType)
                .add("3_MV_003_STATUS", DataTypes.DoubleType)
                .add("3_P_001_STATUS", DataTypes.DoubleType)
                .add("3_P_002_STATUS", DataTypes.DoubleType)
                .add("3_P_003_STATUS", DataTypes.DoubleType)
                .add("3_P_004_STATUS", DataTypes.DoubleType)
                .add("LEAK_DIFF_PRESSURE", DataTypes.DoubleType)
                .add("PLANT_START_STOP_LOG", DataTypes.DoubleType)
                .add("TOTAL_CONS_REQUIRED_FLOW", DataTypes.DoubleType)
                .add("label", DataTypes.IntegerType);



        SparkSession spark = SparkSession
                .builder()
                .appName("batch")
                .config("spark.master", "local")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> df=spark
                .read()
                .option("header",true)
                .format("csv")
                .schema(schema)
                .load("/home/waterquality/IdeaProjects/SparkKafkaConsumer (copy)/src/main/resources/WADI_attackdataLABLE (1).csv");


          // Sonuçları göster
        df.printSchema();
        df.show();
        // Eksik değerleri doldurma veya kaldırma
        Dataset<Row> dfCleaned = df.na().drop();
        df = df.na().fill(0); // Tüm null değerleri 0 yapar.



        //
      //  FIT101	LIT101	 MV101	P101	P102	 AIT201	AIT202	AIT203	FIT201	 MV201	 P201	 P202	P203	 P204	P205	P206	DPIT301	FIT301	LIT301	MV301	MV302	 MV303	MV304	P301	P302	AIT401	AIT402	FIT401	LIT401	P401	P402	P403	P404	UV401	AIT501	AIT502	AIT503	AIT504	FIT501	FIT502	FIT503	FIT504	P501	P502	PIT501	PIT502	PIT503	FIT601	P601	P602	P603	Normal/Attack
        // Assuming encodedDF is your DataFrame
        VectorAssembler assembler = new VectorAssembler()
               // .setInputCols(new String[]{"MV201","LIT101",	 "MV101",	"P101",	"P102",	 "AIT201",	"AIT202","AIT203",	"FIT201",	 "MV201",	 "P201",	 "P202",	"P203",	 "P204",	"P205",	"P206",	"DPIT301",	"FIT301",	"LIT301",	"MV301",	"MV302",	 "MV303",	"MV304",	"P301",	"P302",	"AIT401",	"AIT402",	"FIT401",	"LIT401",	"P401",	"P402",	"P403",	"P404",	"UV401",	"AIT501",	"AIT502",	"AIT503",	"AIT504",	"FIT501",	"FIT502",	"FIT503",	"FIT504",	"P501",	"P502",	"PIT501",	"PIT502",	"PIT503",	"FIT601",	"P601",	"P602",	"P603"})
                .setInputCols(new String[]{
                                "1_AIT_002_PV", "1_FIT_001_PV", "1_LS_001_AL", "1_LS_002_AL", "1_MV_001_STATUS",
                                "1_MV_002_STATUS", "1_MV_003_STATUS", "1_MV_004_STATUS", "1_P_001_STATUS",
                                "1_P_002_STATUS", "1_P_003_STATUS", "1_P_004_STATUS", "1_P_005_STATUS",
                                "1_P_006_STATUS", "2_DPIT_001_PV", "2_FIC_101_CO", "2_FIC_101_PV", "2_FIC_101_SP",
                                "2_FIC_201_CO", "2_FIC_201_PV", "2_FIC_201_SP", "2_FIC_301_CO", "2_FIC_301_PV",
                                "2_FIC_301_SP", "2_FIC_401_CO", "2_FIC_401_PV", "2_FIC_401_SP", "2_FIC_501_CO",
                                "2_FIC_501_PV", "2_FIC_501_SP", "2_FIC_601_CO", "2_FIC_601_PV", "2_FIC_601_SP",
                                "2_FIT_001_PV", "2_FIT_002_PV", "2_FIT_003_PV", "2_FQ_101_PV", "2_FQ_201_PV",
                                "2_FQ_301_PV", "2_FQ_401_PV", "2_FQ_501_PV", "2_FQ_601_PV", "2_LS_001_AL",
                                "2_LS_002_AL", "2_LS_101_AH", "2_LS_101_AL", "2_LS_201_AH", "2_LS_201_AL",
                                "2_LS_301_AH", "2_LS_301_AL", "2_LS_401_AH", "2_LS_401_AL", "2_LS_501_AH",
                                "2_LS_501_AL", "2_LS_601_AH", "2_LS_601_AL", "2_LT_001_PV", "2_LT_002_PV",
                                "2_MCV_007_CO", "2_MCV_101_CO", "2_MCV_201_CO", "2_MCV_301_CO", "2_MCV_401_CO",
                                "2_MCV_501_CO", "2_MCV_601_CO", "2_MV_001_STATUS", "2_MV_002_STATUS",
                                "2_MV_003_STATUS", "2_MV_004_STATUS", "2_MV_005_STATUS", "2_MV_006_STATUS",
                                "2_MV_009_STATUS", "2_MV_101_STATUS", "2_MV_201_STATUS", "2_MV_301_STATUS",
                                "2_MV_401_STATUS", "2_MV_501_STATUS", "2_MV_601_STATUS", "2_P_001_STATUS",
                                "2_P_002_STATUS", "2_P_003_SPEED", "2_P_003_STATUS", "2_P_004_SPEED",
                                "2_P_004_STATUS", "2_PIC_003_CO", "2_PIC_003_PV", "2_PIC_003_SP", "2_PIT_002_PV",
                                "2_PIT_003_PV", "2A_AIT_002_PV", "3_AIT_001_PV", "3_AIT_002_PV", "3_AIT_003_PV",
                                "3_AIT_004_PV", "3_FIT_001_PV", "3_LS_001_AL", "3_LT_001_PV", "3_MV_001_STATUS",
                                "3_MV_002_STATUS", "3_MV_003_STATUS", "3_P_001_STATUS", "3_P_002_STATUS",
                                "3_P_003_STATUS", "3_P_004_STATUS", "LEAK_DIFF_PRESSURE", "PLANT_START_STOP_LOG",
                                "TOTAL_CONS_REQUIRED_FLOW", "label"
                        }
                )
                .setOutputCol("features_vector");



        RandomForestClassifier rfClassifier = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features_vector")
                .setNumTrees(20);

        // Set up the pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rfClassifier});

        // Veri setini eğitim ve test olarak ayırma
        Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3}, 1234L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        // Modeli eğitme
        PipelineModel model = pipeline.fit(trainingData);

        // Modeli kaydetme
        model.write().overwrite().save(MODEL_PATH);
        System.out.println("Model saved at: " + MODEL_PATH);

        // Test verileri ile tahmin
        Dataset<Row> predictions = model.transform(testData);

        // Tahmin sonuçlarını göster
        predictions.select("features_vector", "label", "prediction").show();


        // Performans değerlendirmesi başladı.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy: " + accuracy);

        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = f1Evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1Score);

        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");
        double precision = precisionEvaluator.evaluate(predictions);
        System.out.println("Precision: " + precision);

        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");
        double recall = recallEvaluator.evaluate(predictions);
        System.out.println("Recall: " + recall);

    }
}
