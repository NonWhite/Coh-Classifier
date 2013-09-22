import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CohClassifier {
	private Classifier classifier = null ;
	private DataSource source = null ;
	private Instances data = null ;
	private Evaluation eval = null ;
	
	private static int NUM_ALGORITHMS = 39 ;
	
	private String[] names ;
	private double[][] values ;
	private int archived ;
	
	public CohClassifier(){
		names = new String[ NUM_ALGORITHMS ] ;
		values = new double[ NUM_ALGORITHMS ][ 3 ] ;
		archived = 0 ;
	}
	
	public void setDataFromSingleFile( String path ){
		try{
			source = new DataSource( path ) ;
			data = source.getDataSet() ;
			data.setClassIndex( data.numAttributes() - 1 ) ; 
		}catch( Exception e ){
			System.out.println( "ERROR: No se pudo cargar el archivo: " + path ) ;
		}
	}
	
	public void setClassifier( Classifier C ){
		classifier = C ;
	}
	
	public void build(){
		try{
			classifier.buildClassifier( data ) ;
		}catch( Exception e ){
			System.out.println( "ERROR: No se pudo construir el clasificador " + this.getName() ) ;
		}
	}
	
	public void classify(){
		try{
			eval = new Evaluation( data ) ;
			eval.crossValidateModel( classifier , data , 10 , new Random( 1 ) ) ;
		}catch( Exception e ){
			System.out.println( "ERROR: No se pudo evaluar el clasificador " + this.getName() ) ;
		}
	}
	
	public void export( String path ){
		try{
			File file = new File( path );
			
			if( !file.exists() ) file.createNewFile() ;
 
			FileWriter fw = new FileWriter( file.getAbsoluteFile() ) ;
			BufferedWriter bw = new BufferedWriter( fw ) ;
			bw.write( eval.toMatrixString( "Classified with " + this.getName() ) + "\n" );
			bw.write( eval.toSummaryString( "Summary" , false ) + "\n" ) ;
			bw.close() ;
		}catch( Exception e ){
			System.out.println( "ERROR: No se pudo exportar los datos a la ruta: " + path ) ;
		}
	}

	public void archiveModel(){
		names[ archived ] = this.getName() ;
		values[ archived ][ 0 ] = eval.weightedPrecision() ;
		values[ archived ][ 1 ] = eval.weightedRecall() ;
		values[ archived ][ 2 ] = eval.weightedFMeasure() ;
		archived++ ;
	}
	
	public void toExperimentSummary( String path ){
		sortArchivedModels() ;
		try{
			File file = new File( path ) ;
			if( !file.exists() ) file.createNewFile() ;
			
			FileWriter fw = new FileWriter( file.getAbsoluteFile() ) ;
			PrintWriter pw = new PrintWriter( fw ) ;
			
			String properties[] = { "Precision" , "Recall" , "F-Measure" } ;
			
			pw.write( "SUMMARY\n" );
			pw.printf( "%43s %7s %11s\n" , properties[ 0 ] , properties[ 1 ] , properties[ 2 ] ) ;
			for(int i = 0 ; i < archived ; i++){
				pw.printf( "%-32s %7.2f %9.2f %9.2f\n" ,
						names[ i ] ,
						values[ i ][ 0 ] , values[ i ][ 1 ] , values[ i ][ 2 ] ) ;
			}
			pw.close() ;
		}catch( Exception e ){
			System.out.println( "ERROR: No se pudo exportar los datos a la ruta: " + path ) ;
			e.printStackTrace();
		}
	}
	
	private void sortArchivedModels(){
		for(int i = 0 ; i < archived ; i++){
			for(int j = i + 1 ; j < archived ; j++){
				if( values[ i ][ 2 ] < values[ j ][ 2 ] ){ // Compare F-Measures
					String aux = names[ i ] ;
					names[ i ] = names[ j ] ;
					names[ j ] = aux ;
					
					for(int k = 0 ; k < 3 ; k++){
						double a = values[ i ][ k ] ;
						values[ i ][ k ] = values[ j ][ k ] ;
						values[ j ][ k ] = a ;
					}
				}
			}
		}
	}
	
	public String getName(){
		return classifier.getClass().getSimpleName() ;
	}
}