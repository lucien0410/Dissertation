\documentclass[a4paper]{article}

%% Language and font encodings
\usepackage{gb4e}
\noautomath
\usepackage{natbib}
\usepackage[english]{babel}
\usepackage[utf8x]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{python}

%% Sets page size and margins
%\usepackage[a4paper,top=3cm,bottom=2cm,left=3cm,right=3cm,marginparwidth=1.75cm]{geometry}

%% Useful packages
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[colorinlistoftodos]{todonotes}
\usepackage[colorlinks=true, allcolors=blue]{hyperref}
\usepackage{fixltx2e}
\usepackage{Sweave}

\newcommand*{\myfont}{\fontfamily{ccr}\selectfont}
%the newcommand change the selected word into 'ccr' font
%e.g. \begin{myfont} multi-bleu.perl \end{myfont}

\title{Chapter N: Experimenting Interlinear Glossing Text}
\author{Yuan-Lu Chen}

\begin{document}

\maketitle
\SweaveOpts{concordance=TRUE}
(

\textbf{Assuming that in the previous chapters the following points are addressed already:} 
\begin{itemize}
\item The nature of glosses has been well-explained  (Target audience: CS people without any formal linguistics background):
	\begin{itemize}
	\item What glosses are: A basic intro of interlinear gloss for non-linguists
    \item The golden nature of glosses (encodes NON-LINEAR syntax (i.e. structure parse) and semantics information) 
    \item The potential of gloss:	
		\begin{itemize}
		\item potential: providing disambiguation, labeling important grammar morphemes in the source language, providing morphological analysis, providing one-to-many and many-to-one relations of source tokens and target tokens.  
		\end{itemize}
	\end{itemize}
\item A history of machine translation, and a non-mathy description of the methods of doing machine translation. (Target reader: theoretical linguists)
\end{itemize}


)


\section{Introduction}
The Innovation is to incorporate the gloss information of Interlinear Glossed Text data into machine translation. 

In supervised machine learning models, two factors effects the performance of the trained systems \citep{kotsiantis2007supervised}: a.) the quality of the training data and b.) the choices of the features. The properties of the gloss data as described in *CHAPTERXYZ* make it a better training data than natural language data (Scottish Gaelic in the current case) for the following reasons. First, glosses are more purified that natural language words. The most ideal meaning representation system should be built with mappings where one meaning or function is mapped to one and only one representation. Natural languages fail to do so, given that synonyms and ambiguous words/phrases are ubiquitous in natural languages. On the other hand, Glosses provide this one-to-one mapping. Second, the gloss data provides hierarchical (non-linear) syntactic parsing information to some degree. To determine what the gloss of a word is, linguists have to look for hierarchical context information. 

Therefore, theoretically incorporation of the gloss data should improve the translation systems. Specifically, I propose the following hypothesis:
\begin{exe}  
\ex \textbf{Gloss-helps-hypothesis: the translation systems trained with the gloss data incorporated should outperform the systems trained with only Gaelic and English sentences pairs (i.e. without gloss data).}

The hypothesis can have two versions: strong and weak:
	\begin{xlist}
	\ex \label{strong_hy} Strong version: Gloss may replace the source natural language totally, and the system outperforms the system trained with source natural language to target language sentence pairs (i.e. the baseline systems).   
	\ex \label{weak_hy} Weak version: Gloss only increases the performance of the baseline systems, but cannot replace the source language. 
	\end{xlist}
\end{exe}

The experiments reveal that replacing Gaelic words with glosses doesn't boost up the performance of the translation systems. Thus, the strong version (replacing-Gaelic-with gloss) of the gloss-helps-hypothesis is not attested. However,it is found that if the Gaelic data and the gloss data are combined in a specific way as the training data, the performance of the systems is improved significantly.  

This chapter describes the experiments conducted to test the gloss-helps-hypothesis and the results attest the weak version. 
The rest of the chapter is organized as follows: Section \ref{sec:experimet_setting} describes the constant parameter settings across all the experiments, section \ref{gd_to_gl_to_en} tests the hypothesis in (\ref{strong_hy}), section \ref{gd_plus_gl_to_en} tests the hypothesis in (\ref{weak_hy}),and section 5 concludes the chapter.  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Technical Settings of the Machine Translation Experiments}\label{sec:experimet_setting}
The experiments are conduced by using OpenNMT \citep{2017opennmt}, which implements the state-of-the-art neural net machine translation algorithms \citep{cho2014properties, cho2014learning, bahdanau2014neural}. 
The following default parameters settings of OpenNMT are used across all models so that the only independent variable is the type of the training data: 
\begin{itemize}
\item Word vector size: 500
\item Type of recurrent cell: Long Short Term Memory
\item Number of recurrent layers of the encoder and decoder: 2
\item Number of epochs: 13 
\item Size of mini batch: 64 
\end{itemize}
The data and the scripts will be accessible on GitHub\footnote{\url{https://github.com/lucien0410/Scottish_Gaelic}}, so that the results can be reproduced.   

\section{Gloss Representation Solely Does NOT Outperform Gaelic Sentences} \label{gd_to_gl_to_en}
This section tests the strong version of Gloss-helps-hypothesis in (\ref{strong_hy}).
Given the assumption that gloss may be better than any natural language in terms of representing meanings, it is expected that for neural net machine translation systems it is easier to learn how to translate from the glosses of Scottish Gaelic to English than to learn how to translate from Scottish Gaelic to English. However, the results show that there is no significance difference between the two types of data (i.e. GLOSS $\rightarrow$ English and Gaelic $\rightarrow$ English). 

\subsection{Procedure of the Experiment}
I use repeated random sub-sampling validation to compare the performances of the two type of models. 
Specifically, the samples (i.e. pairs of a gloss line and an English sentence, or pairs of a Gaelic sentence and an English sentence) are randomly split into three datasets: training set (N=6,388), validation set (N=1,000), and test set (N=1,000). The model is trained with the training and validation set (i.e. the model learns how to map the source sequence to the target sequence); the trained model then maps the source sequences of the test set to the predicted target sequences. To evaluate the model, the predicted target sequences are checked against the target sequences of the test set. Specifically, the BLEU score metric \citep{bleu} of each prediction is calculated using \begin{myfont} multi-bleu.perl \end{myfont} 
script, a public implementation of Moses \citep{moses}\footnote{\url{https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl}}.  
This procedure is executed for ten times. 

The procedure is depicted schematically as follows:
\begin{exe}
\ex Definitions of datasets:
\begin{xlist}
\ex Let: \\
    Train, Validation, and Test be sets of random indexes from 0 to 8,387, and \\
    Train $\cap$ Validation $\cap$ Test = $\emptyset$ 

      \begin{xlist}
      \ex \label{GLOSStoENTrain} GLOSStoEN\textsubscript{Train}   = $\{<gloss_i,En_i>  \mid i \in Train \}$ \\
      \ex \label{GDtoENTrain} GDtoEN\textsubscript{Train}   = $\{<GD_i,En_i>  \mid i \in Train \}$ \\
      \ex \label{GLOSStoENVal} GLOSStoEN\textsubscript{Validation}   = $\{<gloss_i,En_i>  \mid i \in Validation \}$ \\
      \ex \label{GDtoENVal} GDtoEN\textsubscript{Validation}   = $\{<GD_i,En_i>  \mid i \in Validation \}$ \\
      \ex \label{GLOSStoENTest}GLOSStoEN\textsubscript{Test} = $\{<gloss_i,En_i>  \mid i \in Test \}$ \\
      \ex \label{GDtoENTest} GDtoEN\textsubscript{Test}    = $\{<GD_i,En_i>  \mid i \in Test \}$ \\
      \ex $\text{score}(MT\textsubscript{GLOSStoEN}, GLOSS\textsubscript{Test}) \geq  \text{score}(MT\textsubscript{GDtoEN}, GD\textsubscript{Test})$
      \end{xlist}
\end{xlist}
\end{exe}


\begin{exe}
\ex Models and Predictions: 
	\begin{xlist}
	\ex Model\textsubscript{GLOSStoEN} = Model trained with (\ref{GLOSStoENTrain}) and (\ref{GLOSStoENVal})
	\ex Predictions\textsubscript{GLOSStoEN} = A list of English sequences that Model\textsubscript{GLOSStoEN} maps to from the gloss sequences in (\ref{GLOSStoENTest}) 
	\ex Model\textsubscript{GDtoEN} = Model trained with (\ref{GDtoENTrain}) and (\ref{GDtoENVal}) 
	\ex Predictions\textsubscript{GDtoEN} = A list of English sequences that Model\textsubscript{GDtoEN} maps to from the gloss sequences in (\ref{GDtoENTest}) 
	\end{xlist}	
\ex Gold-Standard = English sequences in (\ref{GLOSStoENTest}) = English sequences in (\ref{GDtoENTest})
\ex Scores: \\
  \begin{xlist}
	\ex Score\textsubscript{GLOSStoEN} = BLEU(Gold-Standard, Predictions\textsubscript{GLOSStoEN}) \\
	\ex Score\textsubscript{GDtoEN} = BLEU(Gold-Standard, Predictions\textsubscript{GDtoEN}) \\
  \end{xlist}
\end{exe}


\subsection{Result} \label{gdglen_results}
After ten rounds of repeated random sub-sampling validation, ten pairs of scores of the two models are generated, as shown in the following table. 






