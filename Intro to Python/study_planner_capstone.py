# Python-Based Intelligent Study Planner
# Capstone Project

import os
import json
import datetime
from datetime import date
import time

# File to store tasks and performance data
DATA_FILE = "study_planner_data.json"

class StudyPlanner:
    def __init__(self):
        """Initialize the Study Planner."""
        self.tasks = []
        self.subjects = {}
        self.load_data()
    
    def load_data(self):
        """Load existing data from file if available."""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as file:
                    data = json.load(file)
                    self.tasks = data.get('tasks', [])
                    self.subjects = data.get('subjects', {})
                print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Starting with empty data.")
    
    def save_data(self):
        """Save current data to file."""
        try:
            data = {
                'tasks': self.tasks,
                'subjects': self.subjects
            }
            with open(DATA_FILE, 'w') as file:
                json.dump(data, file, indent=4)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_task(self):
        """Add a new study task."""
        try:
            print("\n=== Add New Study Task ===")
            
            # Get task details
            title = input("Enter task title: ")
            
            # Validate and get subject
            subject = input("Enter subject: ")
            
            # Validate and get priority (1-5)
            while True:
                try:
                    priority = int(input("Enter priority (1-5, where 1 is highest): "))
                    if 1 <= priority <= 5:
                        break
                    print("Priority must be between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get deadline
            while True:
                try:
                    deadline_str = input("Enter deadline (YYYY-MM-DD): ")
                    deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d").date()
                    
                    # Check if deadline is not in the past
                    if deadline < date.today():
                        print("Deadline cannot be in the past.")
                        continue
                    break
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")
            
            # Get estimated hours
            while True:
                try:
                    hours = float(input("Enter estimated hours needed: "))
                    if hours <= 0:
                        print("Hours must be positive.")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number.")
            
            # Create task object
            task = {
                'id': len(self.tasks) + 1,
                'title': title,
                'subject': subject,
                'priority': priority,
                'deadline': deadline_str,
                'hours': hours,
                'completed': False,
                'date_added': date.today().isoformat()
            }
            
            # Add task to list
            self.tasks.append(task)
            
            print(f"Task '{title}' added successfully!")
            self.save_data()
            
        except Exception as e:
            print(f"Error adding task: {e}")
    
    def view_tasks(self):
        """View all tasks sorted by priority and deadline."""
        if not self.tasks:
            print("\nNo tasks available.")
            return
        
        print("\n=== Your Study Tasks ===")
        
        # Sort tasks by priority (ascending) and then by deadline
        sorted_tasks = sorted(self.tasks, key=lambda x: (x['priority'], x['deadline']))
        
        for task in sorted_tasks:
            status = "✓" if task['completed'] else "✗"
            deadline = datetime.datetime.strptime(task['deadline'], "%Y-%m-%d").date()
            days_left = (deadline - date.today()).days
            
            # Get deadline status info
            if days_left < 0:
                deadline_info = "OVERDUE"
            elif days_left == 0:
                deadline_info = "TODAY"
            elif days_left == 1:
                deadline_info = "TOMORROW"
            else:
                deadline_info = f"{days_left} days left"
            
            print(f"ID: {task['id']} | {status} | {task['title']} ({task['subject']})")
            print(f"   Priority: {task['priority']} | Deadline: {task['deadline']} ({deadline_info}) | Est. Hours: {task['hours']}")
            print("   " + "-" * 50)
    
    def complete_task(self):
        """Mark a task as completed."""
        self.view_tasks()
        
        if not self.tasks:
            return
        
        try:
            task_id = int(input("\nEnter the ID of the task to mark as completed: "))
            
            # Find the task
            found = False
            for task in self.tasks:
                if task['id'] == task_id:
                    if task['completed']:
                        print("This task is already marked as completed.")
                    else:
                        task['completed'] = True
                        print(f"Task '{task['title']}' marked as completed!")
                        
                        # Update subject performance
                        subject = task['subject']
                        if subject not in self.subjects:
                            self.subjects[subject] = {'completed': 0, 'total': 0}
                        self.subjects[subject]['completed'] += 1
                        
                    found = True
                    break
            
            if not found:
                print("Task not found.")
            else:
                self.save_data()
                
        except ValueError:
            print("Please enter a valid task ID.")
        except Exception as e:
            print(f"Error completing task: {e}")
    
    def add_performance(self):
        """Add a test/quiz score for performance tracking."""
        try:
            print("\n=== Add Performance Score ===")
            
            # Get subject
            subject = input("Enter subject: ")
            
            # Get score
            while True:
                try:
                    score = float(input("Enter your score (0-100): "))
                    if 0 <= score <= 100:
                        break
                    print("Score must be between 0 and 100.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get maximum possible score
            while True:
                try:
                    max_score = float(input("Enter maximum possible score (default 100): ") or "100")
                    if max_score <= 0:
                        print("Maximum score must be positive.")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number.")
            
            # Calculate percentage
            percentage = (score / max_score) * 100
            
            # Initialize subject if not exists
            if subject not in self.subjects:
                self.subjects[subject] = {'scores': [], 'completed': 0, 'total': 0}
            elif 'scores' not in self.subjects[subject]:
                self.subjects[subject]['scores'] = []
            
            # Add score
            self.subjects[subject]['scores'].append({
                'date': date.today().isoformat(),
                'score': score,
                'max_score': max_score,
                'percentage': percentage
            })
            
            print(f"Score added for {subject}: {score}/{max_score} ({percentage:.2f}%)")
            self.save_data()
            
        except Exception as e:
            print(f"Error adding performance: {e}")
    
    def view_performance(self):
        """View performance statistics by subject."""
        if not self.subjects:
            print("\nNo performance data available.")
            return
        
        print("\n=== Performance by Subject ===")
        
        for subject, data in self.subjects.items():
            print(f"\nSubject: {subject}")
            
            # Task completion rate
            if 'completed' in data or 'total' in data:
                completed = data.get('completed', 0)
                total_tasks = sum(1 for task in self.tasks if task['subject'] == subject)
                if total_tasks > 0:
                    completion_rate = (completed / total_tasks) * 100
                    print(f"Task Completion: {completed}/{total_tasks} ({completion_rate:.2f}%)")
                else:
                    print("No tasks recorded for this subject.")
            
            # Score statistics
            if 'scores' in data and data['scores']:
                scores = data['scores']
                percentages = [s['percentage'] for s in scores]
                avg_percentage = sum(percentages) / len(percentages)
                max_percentage = max(percentages)
                min_percentage = min(percentages)
                
                print(f"Average Score: {avg_percentage:.2f}%")
                print(f"Highest Score: {max_percentage:.2f}%")
                print(f"Lowest Score: {min_percentage:.2f}%")
                print(f"Recent Scores:")
                
                # Show recent scores (up to 5)
                for score in sorted(scores, key=lambda x: x['date'], reverse=True)[:5]:
                    print(f"  {score['date']}: {score['score']}/{score['max_score']} ({score['percentage']:.2f}%)")
            else:
                print("No scores recorded for this subject.")
    
    def generate_study_plan(self):
        """Generate a recommended study plan based on priorities and deadlines."""
        if not self.tasks:
            print("\nNo tasks available to create a study plan.")
            return
        
        # Get incomplete tasks
        incomplete_tasks = [task for task in self.tasks if not task['completed']]
        
        if not incomplete_tasks:
            print("\nAll tasks are completed! Good job!")
            return
        
        print("\n=== Recommended Study Plan ===")
        
        # Sort by priority first, then by days remaining
        sorted_tasks = sorted(incomplete_tasks, key=lambda x: (
            x['priority'],
            (datetime.datetime.strptime(x['deadline'], "%Y-%m-%d").date() - date.today()).days
        ))
        
        # Calculate total study hours needed
        total_hours = sum(task['hours'] for task in sorted_tasks)
        
        print(f"\nYou have {len(sorted_tasks)} pending tasks requiring approximately {total_hours:.1f} total hours.")
        
        # Get preferred daily study hours
        try:
            daily_hours = float(input("How many hours can you study per day? "))
            if daily_hours <= 0:
                print("Hours must be positive. Using default of 2 hours.")
                daily_hours = 2
        except ValueError:
            print("Invalid input. Using default of 2 hours.")
            daily_hours = 2
        
        # Calculate days needed
        days_needed = total_hours / daily_hours
        print(f"Based on your availability, you'll need approximately {days_needed:.1f} days to complete all tasks.")
        
        # Generate daily plan
        current_date = date.today()
        current_day_hours = 0
        day_counter = 1
        
        print("\nRecommended Daily Plan:")
        print(f"Day 1 ({current_date.isoformat()}):")
        
        for i, task in enumerate(sorted_tasks, 1):
            task_hours = task['hours']
            deadline = datetime.datetime.strptime(task['deadline'], "%Y-%m-%d").date()
            days_left = (deadline - current_date).days
            
            # If we exceed daily hours, move to next day
            if current_day_hours + task_hours > daily_hours:
                remaining_hours = daily_hours - current_day_hours
                
                if remaining_hours > 0.5:  # If more than 30 minutes left
                    print(f"  - Continue work on {task['title']} ({task['subject']}) - {remaining_hours:.1f} hours")
                    task_hours -= remaining_hours
                
                # Move to next day
                day_counter += 1
                current_date += datetime.timedelta(days=1)
                current_day_hours = 0
                print(f"Day {day_counter} ({current_date.isoformat()}):")
                
                # Add remaining task hours to the next day
                if task_hours > 0:
                    if task_hours <= daily_hours:
                        print(f"  - {task['title']} ({task['subject']}) - {task_hours:.1f} hours" + 
                              f" (Due in {days_left} days)" if days_left > 0 else " (DUE TODAY)")
                        current_day_hours = task_hours
                    else:
                        # This task takes more than a full day
                        print(f"  - Work on {task['title']} ({task['subject']}) - {daily_hours:.1f} hours" + 
                              f" (Due in {days_left} days)" if days_left > 0 else " (DUE TODAY)")
                        current_day_hours = daily_hours
                        
                        # Handle remaining hours on subsequent days (simplified)
                        remaining_task_hours = task_hours - daily_hours
                        print(f"  Note: This task will continue into Day {day_counter+1} ({remaining_task_hours:.1f} more hours needed)")
            else:
                # Add task to current day
                print(f"  - {task['title']} ({task['subject']}) - {task_hours:.1f} hours" + 
                      f" (Due in {days_left} days)" if days_left > 0 else " (DUE TODAY)")
                current_day_hours += task_hours
        
        print("\nNote: This plan prioritizes tasks with higher priority and closer deadlines.")
        print("Adjust as needed based on your actual progress and availability.")
    
    def identify_weak_areas(self):
        """Identify subjects that need more attention based on performance."""
        if not self.subjects:
            print("\nNo performance data available to identify weak areas.")
            return
        
        print("\n=== Areas Needing Improvement ===")
        
        # Analyze each subject
        subject_stats = []
        
        for subject, data in self.subjects.items():
            if 'scores' in data and data['scores']:
                scores = data['scores']
                avg_percentage = sum(s['percentage'] for s in scores) / len(scores)
                
                # Get recent trend (last 3 scores if available)
                recent_scores = sorted(scores, key=lambda x: x['date'], reverse=True)[:3]
                recent_avg = sum(s['percentage'] for s in recent_scores) / len(recent_scores)
                
                # Calculate trend (positive or negative)
                trend = recent_avg - avg_percentage
                
                subject_stats.append({
                    'subject': subject,
                    'average': avg_percentage,
                    'recent_average': recent_avg,
                    'trend': trend,
                    'score_count': len(scores)
                })
        
        # Sort by average score (ascending)
        subject_stats.sort(key=lambda x: x['average'])
        
        if not subject_stats:
            print("No subjects have recorded scores yet.")
            return
        
        # Display weak areas (bottom 50% or below 70%, whichever gives more subjects)
        weak_threshold = min(70, subject_stats[len(subject_stats)//2]['average'] if len(subject_stats) > 1 else 70)
        weak_areas = [s for s in subject_stats if s['average'] < weak_threshold]
        
        if not weak_areas:
            print("Great job! You're performing well in all subjects.")
            return
        
        print("The following subjects may need more attention:\n")
        
        for area in weak_areas:
            trend_str = "improving" if area['trend'] > 2 else "declining" if area['trend'] < -2 else "stable"
            print(f"Subject: {area['subject']}")
            print(f"  Average Score: {area['average']:.2f}%")
            print(f"  Recent Performance: {area['recent_average']:.2f}% ({trend_str})")
            
            # Suggest actions
            if area['trend'] < -5:
                print("  Suggestion: Your performance is dropping significantly. Consider scheduling more time for this subject.")
            elif area['average'] < 60:
                print("  Suggestion: This subject needs immediate attention. Consider seeking additional help.")
            else:
                print("  Suggestion: Regular practice could help improve your understanding.")
            
            print("")
    
    def run(self):
        """Run the main application loop."""
        print("\n****************************************")
        print("*  WELCOME TO INTELLIGENT STUDY PLANNER  *")
        print("****************************************\n")
        
        while True:
            print("\nMAIN MENU:")
            print("1. Add Study Task")
            print("2. View Tasks")
            print("3. Mark Task as Completed")
            print("4. Add Performance Score")
            print("5. View Performance Statistics")
            print("6. Generate Study Plan")
            print("7. Identify Weak Areas")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ")
            
            if choice == '1':
                self.add_task()
            elif choice == '2':
                self.view_tasks()
            elif choice == '3':
                self.complete_task()
            elif choice == '4':
                self.add_performance()
            elif choice == '5':
                self.view_performance()
            elif choice == '6':
                self.generate_study_plan()
            elif choice == '7':
                self.identify_weak_areas()
            elif choice == '8':
                print("\nThank you for using the Intelligent Study Planner! Goodbye!")
                self.save_data()
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
            
            # Pause to let the user read output
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    planner = StudyPlanner()
    planner.run()
