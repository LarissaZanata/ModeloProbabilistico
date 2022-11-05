package naive;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		DataSource ds = new DataSource("src/naive/carro.arff");
		
		Instances ins = ds.getDataSet();
		
		//System.out.println(ins.toString());

		ins.setClassIndex(6); //quem é a classe pra fazer a previsão
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(ins);//cria o classificador
		
		Instance nova = new DenseInstance(7); //num de atributos no total
		
		nova.setDataset(ins); //associação da nova instancia com a base 
		
		nova.setValue(0, "vhigh");
		nova.setValue(1, "vhigh");
		nova.setValue(2, "2");
		nova.setValue(3, "2");
		nova.setValue(4, "small");
		nova.setValue(5, "low");
		
		double probabilidade[] = nb.distributionForInstance(nova);
		System.out.println("unacc: " + probabilidade[0]);
		System.out.println("acc: " + probabilidade[1]);
		System.out.println("good: " + probabilidade[2]);
		System.out.println("vgood: " + probabilidade[3]);
	}

}
