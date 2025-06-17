import logging
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as f
from kedro_component_cartola.utils.auxiliar_functions import back_in_time
from pyspark.sql.types import DoubleType, IntegerType


def gen_financial_features(prod: bool,
                           end_date: str,
                           days_to_evaluate: int,
                           raw_pix: SparkDataFrame,
                           raw_ted: SparkDataFrame,
                           raw_doc: SparkDataFrame
                           ) -> SparkDataFrame:


    
    delta_days = days_to_evaluate + 1
    logging.info(f"Generating financial features for the last {delta_days} days.")
    start_date = back_in_time(end_date, days=delta_days)

    
    groupBy_list = ["nr_cpf_cnpj_sender", "nr_account_sender", "nr_office_sender", "cd_bank_sender",
               "nr_cpf_cnpj_receiver", "nr_account_receiver", "nr_office_receiver", "cd_bank_receiver", "dat_ref_carga"]
    
    pix = (
        raw_pix
        .filter(
            (f.col("dat_ref_carga").between(start_date, end_date))
        )
        .select(
            f.col("nr_cpf_cnpj_sender").cast("string").alias("nr_cpf_cnpj_sender"),
            f.col("nr_account_sender").cast("int").alias("nr_account_sender"),
            f.col("nr_office_sender").cast("int").alias("nr_office_sender"),
            f.col("cd_bank_sender").cast("int").alias("cd_bank_sender"),
            f.col("nr_cpf_cnpj_receiver").cast("string").alias("nr_cpf_cnpj_receiver"),
            f.col("nr_account_receiver").cast("int").alias("nr_account_receiver"),
            f.col("nr_office_receiver").cast("int").alias("nr_office_receiver"),
            f.col("cd_bank_receiver").cast("int").alias("cd_bank_receiver"),
            f.col("vl_pgto").cast("double").alias("vl_titulo"),
            f.col("dat_ref_carga").cast("string").alias("dat_ref_carga")
        )
        .filter(
            (f.col("vl_titulo") > 0)
        )
        .dropDuplicates()
        .groupBy(*groupBy_list)
        .agg(
            f.sum("vl_titulo").alias("vl_titulo"),
            f.count('*').cast('int').alias('qtd_titulo')
        )
    )

    ted = (
        raw_ted
        .filter(
            (f.col("dat_ref_carga").between(start_date, end_date))
        )
        .select(
            f.col("nr_cpf_cnpj_sender").cast("string").alias("nr_cpf_cnpj_sender"),
            f.col("nr_account_sender").cast("int").alias("nr_account_sender"),
            f.col("nr_office_sender").cast("int").alias("nr_office_sender"),
            f.col("cd_bank_sender").cast("int").alias("cd_bank_sender"),
            f.col("nr_cpf_cnpj_receiver").cast("string").alias("nr_cpf_cnpj_receiver"),
            f.col("nr_account_receiver").cast("int").alias("nr_account_receiver"),
            f.col("nr_office_receiver").cast("int").alias("nr_office_receiver"),
            f.col("cd_bank_receiver").cast("int").alias("cd_bank_receiver"),
            f.col("vl_pgto").cast("double").alias("vl_titulo"),
            f.col("dat_ref_carga").cast("string").alias("dat_ref_carga")
        )
        .filter(
            (f.col("vl_titulo") > 0)
        )
        .dropDuplicates()
        .groupBy(*groupBy_list)
        .agg(
            f.sum('vl_titulo').alias('vl_titulo'),
            f.count('*').cast('int').alias('qtd_titulo')
        )
    )

    doc = (
        raw_doc
        .filter(
            (f.col("dat_ref_carga").between(start_date, end_date))
        )
        .select(
            f.col("nr_cpf_cnpj_sender").cast("string").alias("nr_cpf_cnpj_sender"),
            f.col("nr_account_sender").cast("int").alias("nr_account_sender"),
            f.col("nr_office_sender").cast("int").alias("nr_office_sender"),
            f.col("cd_bank_sender").cast("int").alias("cd_bank_sender"),
            f.col("nr_cpf_cnpj_receiver").cast("string").alias("nr_cpf_cnpj_receiver"),
            f.col("nr_account_receiver").cast("int").alias("nr_account_receiver"),
            f.col("nr_office_receiver").cast("int").alias("nr_office_receiver"),
            f.col("cd_bank_receiver").cast("int").alias("cd_bank_receiver"),
            f.col("vl_pgto").cast("double").alias("vl_titulo"),
            f.col("dat_ref_carga").cast("string").alias("dat_ref_carga")
        )
        .filter(
            (f.col("vl_titulo") > 0)
        )
        .dropDuplicates()
        .groupBy(*groupBy_list)
        .agg(
            f.sum('vl_titulo').alias('vl_titulo'),
            f.count('*').cast('int').alias('qtd_titulo')
        )
    )

    # Tabelas de debito automatico, boletos, titulos, seguindo a mesma logica de processamento com alguns filtros e internos
    # Cria um snapshot do periodo de interesse
    financial = pix.union(ted).union(doc)

    logging.info(f"Done processing financial features.")

    return financial


def gen_behavioral_features(prod: bool,
                            end_date: str,
                            days_to_evaluate: int,
                            raw_campanhas: SparkDataFrame,
                            raw_riscos: SparkDataFrame
                            ) -> SparkDataFrame:


    delta_days = days_to_evaluate + 1
    logging.info(f"Generating financial features for the last {delta_days} days.")
    start_date = back_in_time(end_date, days=delta_days)

    
    # Campanhas - 1 so para exemplos, mas poderiam ser mais de uma
    # nr_cpf_cnpj, camp_disparo, camp_conversao, dat_ref_carga

    # Riscos - Ex: 7 dias atraso fatura  -> inclusão bloqueio 80 - target
    # nr_cpf_cnpj, flag_bloqueio, dat_ref_carga

    campanhas = (
        raw_campanhas
        .filter(
            (f.col("dat_ref_carga").between(start_date, end_date))
        )
        .select(
            f.col("nr_cpf_cnpj").cast("string").alias("nr_cpf_cnpj"),
            f.col("camp_disparo").cast("int").alias("camp_disparo"),
            f.col("camp_conversao").cast("int").alias("camp_conversao"),
            f.col("dat_ref_carga").cast("string").alias("dat_ref_carga")
        )
    )

    campanhas = (
        campanhas
        .groupBy("nr_cpf_cnpj", "dat_ref_carga")
        .agg(
            f.sum("camp_disparo").alias("qtd_camp_disparo"),
            f.sum("camp_conversao").alias("qtd_camp_conversao")
        )
    )

    riscos = (
        raw_riscos
        .filter(
            (f.col("dat_ref_carga").between(start_date, end_date))
        )
        .select(
            f.col("nr_cpf_cnpj").cast("string").alias("nr_cpf_cnpj"),
            f.col("flag_bloqueio").cast("int").alias("flag_bloqueio"),
            f.col("dat_ref_carga").cast("string").alias("dat_ref_carga")
        )
    )

    behavioral = (
        campanhas.join(riscos, on=["nr_cpf_cnpj", "dat_ref_carga"], how="full")
        .fillna(0, subset=["qtd_camp_disparo", "qtd_camp_conversao", "flag_bloqueio"])
    )

    behavioral = behavioral.groupBy("nr_cpf_cnpj").agg(
        f.sum('qtd_camp_disparo').alias('qtd_camp_disparo'),
        f.sum('qtd_camp_conversao').alias('qtd_camp_conversao'),
        f.max('flag_bloqueio').alias('flag_bloqueio')
    )

    return behavioral


def agg_financial_features(prod: bool,
                          end_date: str,
                          days_to_evaluate: int,
                          financial_table: SparkDataFrame,
                          cartola_table: SparkDataFrame
                          ) -> SparkDataFrame:

    _POSSIBLE_DELTA_TIME = [1, 7, 15, 30, 60, 90, 180]

    logging.info("Gerando a agregações de transações.")

    financial = financial_table.select(
        "nr_cpf_cnpj_sender", "cd_bank_sender", "nr_cpf_cnpj_receiver",\
        f.col("vl_titulo").alias("vl_tran"),\
        f.col("qtd_titulo").alias("qt_tran"),\
        "dat_ref_carga")
    
    cartola_list = [row.nr_cpf_cnpj for row in cartola_table.filter(f.col("fl_cartola") == 1).select("nr_cpf_cnpj").collect()]

    financial = financial.withColumn("sant_trans", f.when((f.col('cd_bank_sender') == 1), 1).otherwise(0)) \
                     .withColumn("bet_trans", f.when((f.col("nr_cpf_cnpj_receiver").isin(cartola_list)), 1).otherwise(0))
                                   
    financial = financial.drop("cd_bank_sender", "nr_cpf_cnpj_receiver", "dat_ref_carga")

    financial = financial.withColumnRenamed("nr_cpf_cnpj_sender","nr_cpf_cnpj")

    book = financial_table.select(f.col("nr_cpf_cnpj_sender").alias("nr_cpf_cnpj")).distinct()

    for delta_time in _POSSIBLE_DELTA_TIME[:_POSSIBLE_DELTA_TIME.index(days_to_evaluate)]+[days_to_evaluate]:
        start_date = back_in_time(end_date, days=delta_time)

        fin_filt = financial.filter(
            (f.col("dat_ref_carga").between(start_date, end_date))
        )

        sant_bets_filt = ((f.col("sant_trans") == 1) & (f.col("bet_trans") == 1))

        agg = (fin_filt.groupBy("nr_cpf_cnpj").agg(

            f.sum('vl_tran').alias(f"vl_tran_total_sum_{delta_time}"),
            f.sum('qt_tran').alias(f"qt_tran_total_sum_{delta_time}"),

            f.sum(f.when(sant_bets_filt, f.col("vl_tran")).otherwise(0)).alias(f"vl_tran_sant_bets_sum_{delta_time}"),
            f.sum(f.when(sant_bets_filt, f.col("qt_tran")).otherwise(0)).alias(f"qt_tran_sant_bets_sum_{delta_time}"),

            f.max(f.when(sant_bets_filt, f.col("vl_tran")).otherwise(0)).alias(f"vl_tran_sant_bets_max_{delta_time}"),
            f.max(f.when(sant_bets_filt, f.col("qt_tran")).otherwise(0)).alias(f"qt_tran_sant_bets_max_{delta_time}"),

            f.min(f.when(sant_bets_filt, f.col("vl_tran")).otherwise(0)).alias(f"vl_tran_sant_bets_min_{delta_time}"),
            f.min(f.when(sant_bets_filt, f.col("qt_tran")).otherwise(0)).alias(f"qt_tran_sant_bets_min_{delta_time}")
        )).distinct()

        book = book.join(agg, on=['nr_cpf_cnpj'], how='left').fillna(0)

    for i in book.columns:
        if i.startswith("qt"):
            book = book.withColumn(i, f.col(i).cast(IntegerType()))
        if i.startswith("vl"):
            book = book.withColumn(i, f.col(i).cast(DoubleType()))
    
    return book

def master_table(behavioral_features: SparkDataFrame,
                 financial_table: SparkDataFrame) -> SparkDataFrame:
    
    logging.info("Join process..")
    
    return financial_table.join(behavioral_features, on=['nr_cpf_cnpj'], how='inner')