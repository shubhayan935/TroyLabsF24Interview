import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [userName, setUserName] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/recommend', { user_name: userName });
      console.log(response);
      setRecommendations(response.data.recommendations);
      setError('');
    } catch (error) {
      setError('User not found or invalid input.');
      setRecommendations([]);
    }
  };

  return (
    <div className="App">
      <h1>Product Recommendations</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
          placeholder="Enter user name"
        />
        <button type="submit">Get Recommendations</button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <ul>
        {recommendations.map((item, index) => (
          <li key={index}>{item.item}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;
