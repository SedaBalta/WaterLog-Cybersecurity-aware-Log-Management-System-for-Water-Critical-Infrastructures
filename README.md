
# WaterLog: Cybersecurity-aware Log Management System for Water Critical Infrastructures

Water critical infrastructures are increasingly targeted by cyber threats, necessitating robust cybersecurity measures to ensure the continuous and safe delivery of water services. 

This paper presents a comprehensive cybersecurity-aware log management system specifically designed for water critical infrastructures. The system leverages advanced data collection, analysis, and real-time monitoring to detect and mitigate cyber threats effectively. 

## Apache Kafka Running
Firstly, Kafka and Zookeeper are started.

```bash
  sudo systemctl start zookeeper
  sudo systemctl start kafka
```
All operations related to topics on Kafka must be done in the following directory.

```bash
  cd /usr/local/kafka
```

Then, topics are created for communication between producers and consumers.

```bash
  bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic sampleTopic
```

To list the created topics;

```bash
  bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```
If the topics are created successfully, a producer must be created to read the data from the infrastructure in real time.

```bash
  bin/kafka-console-producer.sh --broker-list localhost:9092 --topic sampleTopic
```

Consumers should be created for the real-time distribution of anomaly information from the data passed through the model.

```bash
  bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic sampleTopic --from-beginning
```
Note 1: Sample topic names are used here. A special name must be used for each producer and consumer.

Note 2: If you are reading data from the infrastructure via a wired connection, the relevant IP address must be written instead of "localhost".

## ElasticSearch Installation and Running

```bash
  sudo systemctl start elasticsearch
```

Create a new index on Elasticsearch.

```bash
  curl -X PUT "http://localhost:9200/waterprocess"
```

The "curl" command updates the mapping of an index (waterprocess) on Elasticsearch.

```bash
  curl -X PUT "http://localhost:9200/waterprocess/_mapping" -H 'Content-Type: application/json' -d '
  {
  "properties": {
    "Treat_ScenarioID": { "type": "double" },
    "Treat_Level2": { "type": "double" },
    "Treat_Level1": { "type": "double" },
    "Treat_CurrentVolume": { "type": "double" },
    "Dam_Level": { "type": "double" },
    "Dam_CurrentVolume": { "type": "double" },
    "Treat_PumpFlowLiterMin": { "type": "double" },
    "Treat_Pump3": { "type": "keyword" },
    "Treat_Pump2": { "type": "keyword" },
    "Treat_Pump1": { "type": "keyword" },
    "Treat_LimitSwitch1": { "type": "keyword" },
    "Dam_Pump3": { "type": "keyword" },
    "Dam_Pump2": { "type": "keyword" },
    "Dam_Pump1": { "type": "keyword" },
    "Dam_LimitSwitch": { "type": "keyword" },
    "Dam_Chlor_Raw": { "type": "double" },
    "label": { "type": "double" },
    "Treat_Pump3Index": { "type": "double" },
    "Treat_Pump2Index": { "type": "double" },
    "Treat_Pump1Index": { "type": "double" },
    "Treat_LimitSwitch1Index": { "type": "double" },
    "Dam_Pump3Index": { "type": "double" },
    "Dam_Pump2Index": { "type": "double" },
    "Dam_Pump1Index": { "type": "double" },
    "Dam_LimitSwitchIndex": { "type": "double" },
    "prediction": { "type": "double" }
  }
}'
```

From localhost:5601, write data/index management to the search section.

then from the left panel, Kibana Index patterns are recreated with the same index name.
The visualization of the data from the dashboard was done via localhost:5601.
The relevant application file was implemented on elasticsearch.java in the "elk" folder.

## Kibana Installation 

```bash
  sudo apt install kibana
  sudo systemctl enable kibana
```

For starting

```bash
  sudo systemctl start kibana
```

## Nginx Installation 

```bash
  sudo apt install nginx
  sudo ufw allow 'Nginx HTTP'
  sudo ufw enable
  systemctl status nginx
```

## WaterLog Dataset Access

You can access the dataset from the drive link below.

https://drive.google.com/file/d/1j7i8MJEXHM2yXYdXZ9g7rEDgfPhJB7Wd/view?usp=sharing

## How do I run the project?

Install the dependencies in the pom.xml file located in the SparkKafkaConsumer folder.

Then, after the above installations and work are completed, use the files located in the src/main/java directory.

batch.java -> Used for training.
realtime.java -> Used for real-time data flow and anomaly distribution over the recorded model. The myCustomFunc() function performs distribution by filtering over two different topics.

realtimeListenersdelay -> Used to analyze the performance of the system and calculate the delay times between batches.

Other performance tests are performed via the Spark web port (localhost:4040).