import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
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
	
	public CohClassifier(){
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
	
	public String getName(){
		return classifier.getClass().getSimpleName() ;
	}
}
