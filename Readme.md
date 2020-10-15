LINEAR REGRESSION TO PERFORM NODE ANALYSIS IN AN IOT SYSTEM

Folder contains the following:
- The main program which is the python file "main.py".
- The database holding the trust values of all 5 parameters called "node_database.xlsx".
- The database holding the list of working, malicious and potentially malicious nodes.
- The screenshots folder containing images of graphs and tables which come as outputs of the program

The process:
- Node_Analyser analyses trust values of 50 nodes using 5 parameters from the node_database
- It then displays the predicted values of all 5 parameters and the resultant trust factor for all 50 nodes (Only for representational purposes)
- Displays a graph to show how linear regression works
- Displays another graph that shows the trends in the trust parameters and the resultant trust factor over the no. of nodes in the system
- Analyses the data and classifies the nodes into working, malicious and potentially malicious and writes them to the malicious_node_data database 
- Takes the actual values of the parameters as input and stores them in the node_database
- Once it says 'DATABASE UPDATED', the complete operation has finished
- The program can be run repeatedly given that the databases have not been manually tampered with.

Uses:
- To predict the next trust factor of a particular node.
- To analyse the effects on the trust value of a node when it turns malicious or dead. (Can detect a bad node within 1-2 iterations)
- To classify the nodes into working, malicious and potentially malicious nodes.
