package naive;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args){
		final String BASE_ORIGINAL = "src/naive/carro.arff";
		final String BASE_TREINAMENTO = "src/naive/base_car_treino.arff";
		final String BASE_VALIDACAO = "src/naive/base_car_validacao.arff";
		final String BASE_TESTE = "src/naive/base_car_testeFinal.arff";
		
		try {
			naiveBayes(BASE_TREINAMENTO, "Treinamento", 60);
			naiveBayes(BASE_VALIDACAO, "Validação", 20);
			naiveBayes(BASE_TESTE, "Teste Final", 1);
		} catch (Throwable e) {
			e.printStackTrace();
		}
		
	}
	/*
	 * O parametro qtd é a quantidade de vezes que irá executar o Naive Bayes para cada dataSet
	 * */
	public static void naiveBayes(String nomeBase, String op, int qtd) throws Throwable {
		
		int i = 0;
		while(i < qtd) {
			DataSource ds = new DataSource(nomeBase);
			
			Instances ins = ds.getDataSet();
			
			//System.out.println(ins.toString());

			ins.setClassIndex(6); 
			
			NaiveBayes nb = new NaiveBayes();
			
			nb.buildClassifier(ins);//cria o classificador
			
			Instance nova = new DenseInstance(7); //quantidade de atributos no total
			
			nova.setDataset(ins); //associação da nova instancia com a base 
			
			nova.setValue(0, "vhigh"); //Preço total
			nova.setValue(1, "high"); // Preço de compra
			nova.setValue(2, "2"); //Quantidade de portas 
			nova.setValue(3, "4"); //Quantidade de pessoa
			nova.setValue(4, "small"); //Porta-malas pequeno
			nova.setValue(5, "low");  //Baixa segurança
			
			
			double probabilidade[] = nb.distributionForInstance(nova);
			
			if(i == qtd-1) {
				System.out.println("Resultados para a base de " + op + ":");
				System.out.println("Probabilidade de ocorrer unacc: " + probabilidade[0]);
				System.out.println("Probabilidade de ocorrer acc: " + probabilidade[1]);
				System.out.println("Probabilidade de ocorrer good: " + probabilidade[2]);
				System.out.println("Probabilidade de ocorrer vgood: " + probabilidade[3]);
				System.out.println(" ");
			}
			i++;
		}
	}

}
