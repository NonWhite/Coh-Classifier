import weka.classifiers.Classifier;

public class Main {

	public static void main(String[] args) {
		try{
			test() ;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void test(){
		Classifier CL[] = new Classifier[ 100 ] ;
		CL[ 0 ] = new weka.classifiers.bayes.BayesNet() ;
		CL[ 1 ] = new weka.classifiers.bayes.NaiveBayes() ;
		CL[ 2 ] = new weka.classifiers.bayes.NaiveBayesMultinomial() ;
		CL[ 3 ] = new weka.classifiers.bayes.NaiveBayesMultinomialText() ;
		CL[ 4 ] = new weka.classifiers.bayes.NaiveBayesMultinomialUpdateable() ;
		CL[ 5 ] = new weka.classifiers.bayes.NaiveBayesUpdateable() ;
		CL[ 6 ] = new weka.classifiers.functions.Logistic() ;
		CL[ 7 ] = new weka.classifiers.functions.MultilayerPerceptron() ;
		CL[ 8 ] = new weka.classifiers.functions.SimpleLogistic() ;
		CL[ 9 ] = new weka.classifiers.functions.SMO() ;
		CL[ 10 ] = new weka.classifiers.lazy.IBk() ;
		CL[ 11 ] = new weka.classifiers.lazy.KStar() ;
		CL[ 12 ] = new weka.classifiers.lazy.LWL() ;
		CL[ 13 ] = new weka.classifiers.meta.AdaBoostM1() ;
		CL[ 14 ] = new weka.classifiers.meta.AttributeSelectedClassifier() ;
		CL[ 15 ] = new weka.classifiers.meta.Bagging() ;
		CL[ 16 ] = new weka.classifiers.meta.ClassificationViaRegression() ;
		CL[ 17 ] = new weka.classifiers.meta.CVParameterSelection() ;
		CL[ 18 ] = new weka.classifiers.meta.FilteredClassifier() ;
		CL[ 19 ] = new weka.classifiers.meta.LogitBoost() ;
		CL[ 20 ] = new weka.classifiers.meta.MultiClassClassifier() ;
		CL[ 21 ] = new weka.classifiers.meta.MultiClassClassifierUpdateable() ;
		CL[ 22 ] = new weka.classifiers.meta.MultiScheme() ;
		CL[ 23 ] = new weka.classifiers.meta.RandomCommittee() ;
		CL[ 24 ] = new weka.classifiers.meta.RandomSubSpace() ;
		CL[ 25 ] = new weka.classifiers.meta.Stacking() ;
		CL[ 26 ] = new weka.classifiers.meta.Vote() ;
		CL[ 27 ] = new weka.classifiers.rules.DecisionTable() ;
		CL[ 28 ] = new weka.classifiers.rules.JRip() ;
		CL[ 29 ] = new weka.classifiers.rules.OneR() ;
		CL[ 30 ] = new weka.classifiers.rules.PART() ;
		CL[ 31 ] = new weka.classifiers.rules.ZeroR() ;
		CL[ 32 ] = new weka.classifiers.trees.DecisionStump() ;
		CL[ 33 ] = new weka.classifiers.trees.HoeffdingTree() ;
		CL[ 34 ] = new weka.classifiers.trees.J48() ;
		CL[ 35 ] = new weka.classifiers.trees.LMT() ;
		CL[ 36 ] = new weka.classifiers.trees.RandomForest() ;
		CL[ 37 ] = new weka.classifiers.trees.RandomTree() ;
		CL[ 38 ] = new weka.classifiers.trees.REPTree() ;
		
		CohClassifier coh = new CohClassifier() ;
		for(int i = 0 ; i < 39 ; i++){
//			coh = new CohClassifier() ;
			coh.setClassifier( CL[ i ] ) ;
			coh.setDataFromSingleFile( "/home/nonwhite/data.arff" ) ;
			coh.build() ;
			coh.classify() ;
			coh.export( "/home/nonwhite/runs/" + (i+1) + " - " + coh.getName() + ".txt" ) ;
			coh.archiveModel() ;
		}
		coh.toExperimentSummary( "/home/nonwhite/runs/summary.txt" ) ;
	}
}
