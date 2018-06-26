using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Models;
// 0.1.0 at 9.5.2018 :-)
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace statePred
{
    class Program
    {
        // STEP 1: Define your data structures
        // use a-incidents
        //  "number","caller_id","category","priority","incident_state","short_description","assigned_to","sys_updated_on","sys_updated_by","assignment_group","company"
     
        static void Main(string[] args)
        {
            Console.WriteLine("ML for Elisa!");
            const string modelpath = @".\Data\Model.zip";

            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of dataPath is set to 'Copy always'
            string dataPath = "data/incidents.txt";
            pipeline.Add(new TextLoader<IncidentData>(dataPath, separator: ","));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));            

            // Puts all features into a vector  (ColumnConcatenator is ok for numbers)
            pipeline.Add(new TextFeaturizer("Features", "ShortDescription"));


            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline.  StochasticDualCoordinateAscentClassifier, , LogisticRegressor
            // This is a classification scenario (Question: What type of state is this?)
            pipeline.Add(new NaiveBayesClassifier()); //FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            
            // Convert the Label back into original text (after converting to number in step 3)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<IncidentData, IncidentCategoryPrediction>();

            model.WriteAsync(modelpath);

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions
            Console.WriteLine();
            Console.WriteLine("PredictionModel evaluation");
            Console.WriteLine("---------------------------------------------------------------------");

            IEnumerable<IncidentData> incidents = new[]
            {
                new IncidentData()
                {
                    ShortDescription = "Intraan ei pääsee",
                    Label = ""
                } 
            };

            
            foreach (var item in incidents)
            {
                Console.WriteLine($"Incident \"{item.ShortDescription}\" | Category predicted: {model.Predict(item).PredictedLabels}");                
            }
            

            //STEP 7 Evaluate your model
            var evaluator = new ClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, new TextLoader<IncidentData>(dataPath, separator: ","));
            
            Console.WriteLine("-----------------------------------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.AccuracyMacro:P2}");
            Console.WriteLine("-----------------------------------------------------------------------");
        }
    }
}
