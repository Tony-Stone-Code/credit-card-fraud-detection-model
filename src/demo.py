"""
Demo script for testing the fraud detection API.
Shows various usage examples and test cases.
"""
import requests
import json
import time
import random
from typing import Dict, List


class FraudDetectionDemo:
    """Demo client for fraud detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize demo client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        
    def test_health(self):
        """Test health check endpoint."""
        print("\n" + "="*60)
        print("1. TESTING HEALTH CHECK")
        print("="*60)
        
        response = requests.get(f"{self.base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    
    def test_model_info(self):
        """Test model info endpoint."""
        print("\n" + "="*60)
        print("2. TESTING MODEL INFO")
        print("="*60)
        
        response = requests.get(f"{self.api_url}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    
    def test_single_prediction(self):
        """Test single transaction prediction."""
        print("\n" + "="*60)
        print("3. TESTING SINGLE PREDICTION")
        print("="*60)
        
        # Example transaction (likely non-fraud)
        transaction = {
            "Time": 406,
            "Amount": 150.50,
            "V1": -1.3598071336738,
            "V2": -0.0727811733098497,
            "V3": 2.53634673796914,
            "V4": 1.37815522427443,
            "V5": -0.338320769942518,
            "V6": 0.462387777762292,
            "V7": 0.239598554061257,
            "V8": 0.0986979012610507,
            "V9": 0.363786969611213,
            "V10": 0.0907941719789316,
            "V11": -0.551599533260813,
            "V12": -0.617800855762348,
            "V13": -0.991389847235408,
            "V14": -0.311169353699879,
            "V15": 1.46817697209427,
            "V16": -0.470400525259478,
            "V17": 0.207971241929242,
            "V18": 0.0257905801985591,
            "V19": 0.403992960255733,
            "V20": 0.251412098239705,
            "V21": -0.018306777944153,
            "V22": 0.277837575558899,
            "V23": -0.110473910188767,
            "V24": 0.0669280749146731,
            "V25": 0.128539358273528,
            "V26": -0.189114843888824,
            "V27": 0.133558376740387,
            "V28": -0.0210530531905623
        }
        
        print(f"Transaction Amount: ${transaction['Amount']}")
        
        response = requests.post(
            f"{self.api_url}/predict",
            json=transaction
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Highlight key results
        pred = result['prediction']
        print(f"\n{'🚨 FRAUD DETECTED' if pred['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"Fraud Probability: {pred['fraud_probability']:.4f}")
        print(f"Risk Score: {pred['risk_score']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        
        return result
    
    def test_batch_prediction(self):
        """Test batch prediction with multiple transactions."""
        print("\n" + "="*60)
        print("4. TESTING BATCH PREDICTION")
        print("="*60)
        
        # Generate sample transactions
        transactions = []
        for i in range(5):
            transaction = self._generate_sample_transaction(i)
            transactions.append(transaction)
        
        batch_request = {
            "transactions": transactions
        }
        
        print(f"Sending {len(transactions)} transactions...")
        
        response = requests.post(
            f"{self.api_url}/predict/batch",
            json=batch_request
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        print(f"Batch ID: {result['batch_id']}")
        print(f"Total Transactions: {result['total_transactions']}")
        print(f"Total Latency: {result['total_latency_ms']:.2f}ms")
        print(f"Avg Latency per Transaction: {result['total_latency_ms']/result['total_transactions']:.2f}ms")
        
        # Show summary
        fraud_count = sum(1 for p in result['predictions'] if p['prediction']['is_fraud'])
        print(f"\nResults Summary:")
        print(f"  Fraudulent: {fraud_count}")
        print(f"  Legitimate: {len(transactions) - fraud_count}")
        
        # Show details for each prediction
        print(f"\nDetailed Results:")
        for i, pred_result in enumerate(result['predictions']):
            pred = pred_result['prediction']
            status = "🚨 FRAUD" if pred['is_fraud'] else "✅ LEGIT"
            print(f"  Transaction {i+1}: {status} | Probability: {pred['fraud_probability']:.4f} | Risk: {pred['risk_score']}")
        
        return result
    
    def test_feature_importance(self):
        """Test feature importance endpoint."""
        print("\n" + "="*60)
        print("5. TESTING FEATURE IMPORTANCE")
        print("="*60)
        
        response = requests.get(f"{self.api_url}/model/features")
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(result['top_10_features'].items(), 1):
            print(f"  {i}. {feature}: {importance:.6f}")
        
        return result
    
    def test_performance(self, num_requests: int = 100):
        """Test API performance with multiple requests."""
        print("\n" + "="*60)
        print(f"6. PERFORMANCE TEST ({num_requests} requests)")
        print("="*60)
        
        transaction = self._generate_sample_transaction(0)
        
        latencies = []
        errors = 0
        
        print("Sending requests...")
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                req_start = time.time()
                response = requests.post(
                    f"{self.api_url}/predict",
                    json=transaction
                )
                latency = (time.time() - req_start) * 1000
                latencies.append(latency)
                
                if response.status_code != 200:
                    errors += 1
                    
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{num_requests} requests...")
                    
            except Exception as e:
                errors += 1
                print(f"  Error on request {i + 1}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        throughput = num_requests / total_time
        
        print(f"\nPerformance Results:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} requests/second")
        print(f"  Avg Latency: {avg_latency:.2f}ms")
        print(f"  Min Latency: {min_latency:.2f}ms")
        print(f"  Max Latency: {max_latency:.2f}ms")
        print(f"  Errors: {errors}")
        print(f"  Success Rate: {((num_requests - errors) / num_requests * 100):.2f}%")
        
        return {
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "errors": errors
        }
    
    def _generate_sample_transaction(self, seed: int = 0) -> Dict:
        """Generate a sample transaction for testing."""
        random.seed(seed)
        
        return {
            "Time": random.randint(0, 172800),  # Up to 48 hours
            "Amount": round(random.uniform(1, 1000), 2),
            "V1": random.uniform(-3, 3),
            "V2": random.uniform(-3, 3),
            "V3": random.uniform(-3, 3),
            "V4": random.uniform(-3, 3),
            "V5": random.uniform(-3, 3),
            "V6": random.uniform(-3, 3),
            "V7": random.uniform(-3, 3),
            "V8": random.uniform(-3, 3),
            "V9": random.uniform(-3, 3),
            "V10": random.uniform(-3, 3),
            "V11": random.uniform(-3, 3),
            "V12": random.uniform(-3, 3),
            "V13": random.uniform(-3, 3),
            "V14": random.uniform(-3, 3),
            "V15": random.uniform(-3, 3),
            "V16": random.uniform(-3, 3),
            "V17": random.uniform(-3, 3),
            "V18": random.uniform(-3, 3),
            "V19": random.uniform(-3, 3),
            "V20": random.uniform(-3, 3),
            "V21": random.uniform(-3, 3),
            "V22": random.uniform(-3, 3),
            "V23": random.uniform(-3, 3),
            "V24": random.uniform(-3, 3),
            "V25": random.uniform(-3, 3),
            "V26": random.uniform(-3, 3),
            "V27": random.uniform(-3, 3),
            "V28": random.uniform(-3, 3),
        }
    
    def run_all_tests(self):
        """Run all demo tests."""
        print("\n" + "="*60)
        print("FRAUD DETECTION API - COMPREHENSIVE DEMO")
        print("="*60)
        
        try:
            # Test 1: Health check
            self.test_health()
            
            # Test 2: Model info
            self.test_model_info()
            
            # Test 3: Single prediction
            self.test_single_prediction()
            
            # Test 4: Batch prediction
            self.test_batch_prediction()
            
            # Test 5: Feature importance
            self.test_feature_importance()
            
            # Test 6: Performance test
            self.test_performance(num_requests=50)
            
            print("\n" + "="*60)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
            print("="*60)
            
        except requests.exceptions.ConnectionError:
            print("\n❌ ERROR: Could not connect to API server")
            print("Make sure the server is running at http://localhost:8000")
            print("Start the server with: python src/api_server.py")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            raise


def main():
    """Run demo tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo and test fraud detection API')
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='API base URL'
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['health', 'info', 'single', 'batch', 'features', 'performance', 'all'],
        default='all',
        help='Which test to run'
    )
    parser.add_argument(
        '--num-requests',
        type=int,
        default=50,
        help='Number of requests for performance test'
    )
    
    args = parser.parse_args()
    
    demo = FraudDetectionDemo(args.url)
    
    if args.test == 'all':
        demo.run_all_tests()
    elif args.test == 'health':
        demo.test_health()
    elif args.test == 'info':
        demo.test_model_info()
    elif args.test == 'single':
        demo.test_single_prediction()
    elif args.test == 'batch':
        demo.test_batch_prediction()
    elif args.test == 'features':
        demo.test_feature_importance()
    elif args.test == 'performance':
        demo.test_performance(args.num_requests)


if __name__ == '__main__':
    main()
