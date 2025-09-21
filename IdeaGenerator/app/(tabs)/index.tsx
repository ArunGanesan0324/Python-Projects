import React, { useState } from 'react';
import { View, Text, Button, Alert, StyleSheet } from 'react-native';
import { Picker } from '@react-native-picker/picker';

const App: React.FC = () => {
  const [domain, setDomain] = useState<string>('');
  const [difficulty, setDifficulty] = useState<string>('');
  const [prediction, setPrediction] = useState<string[] | null>(null);

  const handleSubmit = async (): Promise<void> => {
    if (!domain || !difficulty) {
      Alert.alert('Error', 'Please select both domain and difficulty');
      console.log('Error: Domain or difficulty not selected');
      return;
    }

    console.log('Sending request with domain:', domain, 'and difficulty:', difficulty);

    try {
      const response = await fetch('http://192.168.29.33:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ domain, difficulty }),
      });

      console.log('Response status:', response.status);

      const data: { prediction?: string } = await response.json();

      console.log('Response data:', data);

      if (data.prediction) {
        setPrediction(data.prediction.split('\n'));
        console.log('Prediction received:', data.prediction);
      } else {
        Alert.alert('No Results', 'No project ideas found for the selected domain and difficulty.');
        console.log('No results found');
      }
    } catch (error) {
      Alert.alert('Error', `Failed to fetch predictions. Ensure backend is running. ${error}`);
      console.log('Error fetching predictions:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Select Domain:</Text>
      <View style={styles.pickerContainer}>
        <Picker selectedValue={domain} onValueChange={(itemValue) => setDomain(itemValue)} style={styles.picker}>
          <Picker.Item label="Select a domain" value="" />
          <Picker.Item label="AI/ML" value="AI/ML" />
          <Picker.Item label="Big Data" value="Big Data" />
          <Picker.Item label="Blockchain" value="Blockchain" />
          <Picker.Item label="Cloud Computing" value="Cloud Computing" />
          <Picker.Item label="Computer Vision" value="Computer Vision" />
          <Picker.Item label="Cybersecurity" value="Cybersecurity" />
          <Picker.Item label="Data Science" value="Data Science" />
          <Picker.Item label="Embedded Systems" value="Embedded Systems" />
          <Picker.Item label="Game Development" value="Game Development" />
          <Picker.Item label="IoT" value="IoT" />
          <Picker.Item label="Networking" value="Networking" />
          <Picker.Item label="Web Development" value="Web Development" />
        </Picker>
      </View>

      <Text style={styles.label}>Select Difficulty:</Text>
      <View style={styles.pickerContainer}>
        <Picker selectedValue={difficulty} onValueChange={(itemValue) => setDifficulty(itemValue)} style={styles.picker}>
          <Picker.Item label="Select difficulty" value="" />
          <Picker.Item label="Beginner" value="Beginner" />
          <Picker.Item label="Intermediate" value="Intermediate" />
          <Picker.Item label="Advanced" value="Advanced" />
        </Picker>
      </View>

      <Button title="Submit" onPress={handleSubmit} color="#007BFF" />

      {prediction && Array.isArray(prediction) && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Suggested Project Ideas:</Text>
          {prediction.map((idea, index) => (
            <Text key={index} style={styles.resultText}>{index + 1}. {idea}</Text>
          ))}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
    backgroundColor: '#f8f9fa',
  },
  label: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  pickerContainer: {
    backgroundColor: '#fff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
    marginBottom: 15,
  },
  picker: {
    height: 50,
    width: '100%',
  },
  resultContainer: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#fff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#007BFF',
  },
  resultText: {
    fontSize: 16,
    color: '#333',
  },
});

export default App;
