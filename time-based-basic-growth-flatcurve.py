import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from datetime import datetime, timedelta

class Distribution:
    def __init__(self, total_units=100, num_epochs=7, epoch_duration=timedelta(days=1)):
        self.total_units = total_units
        self.num_epochs = num_epochs
        self.epoch_duration = epoch_duration
        self.total_seconds = self.num_epochs * self.epoch_duration.total_seconds()
        self.linear_epochs = 2  # Number of epochs with linear growth
        self.create_distribution_function()
        self.distributed_units = 0
        self.distribution_data = []
        self.last_distribution_time = 0

    def create_distribution_function(self):
        def combined_function(x):
            if x <= self.linear_epochs:
                # Linear function for the first two epochs with adjustable steepness
                steepness = 34
                return steepness * x * (1 / self.linear_epochs)
            else:
                # Quadratic function for the rest, shifted to start at the end of the linear part
                a = 1
                b = -2 * a * (self.num_epochs - self.linear_epochs)
                c = a * (self.num_epochs - self.linear_epochs) ** 2
                return a * (x - self.linear_epochs) ** 2 + b * (x - self.linear_epochs) + c + 1

        self.distribution_function = np.vectorize(combined_function)

        # Normalize the function so the area under the curve is total_units
        area, _ = integrate.quad(self.distribution_function, 0, self.num_epochs)
        self.normalization_factor = self.total_units / area

    def get_distribution_rate_at_time(self, time):
        epoch_time = time / self.epoch_duration.total_seconds()
        return self.distribution_function(epoch_time) * self.normalization_factor / self.epoch_duration.total_seconds()

    def trigger_distribution(self, current_time):
        if current_time > self.total_seconds or self.distributed_units >= self.total_units:
            return 0

        time_since_last = current_time - self.last_distribution_time
        avg_rate = (self.get_distribution_rate_at_time(self.last_distribution_time) + 
                    self.get_distribution_rate_at_time(current_time)) / 2

        distribution = avg_rate * time_since_last
        distribution = min(distribution, self.total_units - self.distributed_units)
        distribution = round(distribution, 2)

        self.distributed_units += distribution
        self.distribution_data.append((current_time, self.distributed_units))
        self.last_distribution_time = current_time

        return distribution

def run_simulation(tokens_to_distribute = 100, num_epochs=7, epoch_duration=timedelta(days=1)):
    dist = Distribution(total_units=tokens_to_distribute, num_epochs=num_epochs, epoch_duration=epoch_duration)
    triggers = sorted(np.random.uniform(0, dist.total_seconds, 1000))  # 1000 random triggers

    for time in triggers:
        amount = dist.trigger_distribution(time)
        if amount > 0:
            epoch = time / dist.epoch_duration.total_seconds()
            print(f"Time: {epoch:.2f} epochs, Distributed: {amount:.2f}, Total: {dist.distributed_units:.2f}")

    print(f"\nTotal distributed: {dist.distributed_units:.2f}")
    print(f"Remaining: {dist.total_units - dist.distributed_units:.2f}")

    return dist

def plot_distribution(dist):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot the distribution function
    x = np.linspace(0, dist.num_epochs, 1000)
    y = [dist.get_distribution_rate_at_time(t * dist.epoch_duration.total_seconds()) * dist.epoch_duration.total_seconds() for t in x]
    ax1.plot(x, y)
    ax1.set_title('Distribution Rate Over Time')
    ax1.set_xlabel('Time (epochs)')
    ax1.set_ylabel('Distribution Rate (units/epoch)')
    ax1.grid(True)

    # Add vertical line to show transition from linear to quadratic
    ax1.axvline(x=dist.linear_epochs, color='r', linestyle='--', alpha=0.5)
    ax1.text(dist.linear_epochs, ax1.get_ylim()[1], 'Linear to Quadratic Transition', 
             rotation=90, va='top', ha='right', alpha=0.5)

    # Add labels every quarter epoch for distribution rate
    for i in range(0, dist.num_epochs * 4 + 1):
        time = i / 4
        if time <= dist.num_epochs:
            rate = dist.get_distribution_rate_at_time(time * dist.epoch_duration.total_seconds()) * dist.epoch_duration.total_seconds()
            ax1.annotate(f'{rate:.2f}', (time, rate), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Add epoch summary for distribution rate
    for epoch in range(dist.num_epochs):
        epoch_rate = sum([dist.get_distribution_rate_at_time((epoch + h/24) * dist.epoch_duration.total_seconds()) * dist.epoch_duration.total_seconds() for h in range(24)]) / 24
        ax1.annotate(f'Epoch {epoch+1}\n{epoch_rate:.2f}', (epoch + 0.5, ax1.get_ylim()[1]), ha='center', va='top', fontsize=9)

    # Plot the cumulative distribution
    times, amounts = zip(*dist.distribution_data)
    times = [t / dist.epoch_duration.total_seconds() for t in times]  # Convert to epochs
    ax2.plot(times, amounts, '-')
    ax2.set_title('Cumulative Distribution Over Time')
    ax2.set_xlabel('Time (epochs)')
    ax2.set_ylabel('Cumulative Units Distributed')
    ax2.grid(True)

    # Add vertical line to show transition from linear to quadratic
    ax2.axvline(x=dist.linear_epochs, color='r', linestyle='--', alpha=0.5)
    ax2.text(dist.linear_epochs, ax2.get_ylim()[1], 'Linear to Quadratic Transition', 
             rotation=90, va='top', ha='right', alpha=0.5)

    # Add labels every quarter epoch for cumulative distribution
    labeled_times = np.arange(0, dist.num_epochs + 0.25, 0.25)  # Every quarter epoch
    for time in labeled_times:
        if time <= max(times):
            amount = np.interp(time, times, amounts)
            ax2.annotate(f'{amount:.2f}', (time, amount), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Add epoch summary for cumulative distribution
    for epoch in range(dist.num_epochs):
        epoch_amount = np.interp(epoch + 1, times, amounts) - np.interp(epoch, times, amounts)
        total_amount = np.interp(epoch + 1, times, amounts)
        print(f'Epoch {epoch+1}: +{epoch_amount:.2f}, Total: {total_amount:.2f}')
        ax2.annotate(f'Epoch {epoch+1}\n+{epoch_amount:.2f}\nTotal: {total_amount:.2f}', 
                     (epoch + 0.5, ax2.get_ylim()[1]), ha='center', va='top', fontsize=9)

    # Plot bar chart for amount distributed per epoch
    epoch_amounts = [np.interp(epoch + 1, times, amounts) - np.interp(epoch, times, amounts) for epoch in range(dist.num_epochs)]
    ax3.bar(range(1, dist.num_epochs + 1), epoch_amounts)
    ax3.set_title('Amount Distributed per Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Units Distributed')
    ax3.grid(True, axis='y')

    # Add value labels on top of each bar
    for i, v in enumerate(epoch_amounts):
        ax3.text(i + 1, v, f'{v:.2f}', ha='center', va='bottom')

    # Add vertical line to show transition from linear to quadratic
    ax3.axvline(x=dist.linear_epochs + 0.5, color='r', linestyle='--', alpha=0.5)
    ax3.text(dist.linear_epochs + 0.5, ax3.get_ylim()[1], 'Linear to Quadratic Transition', 
             rotation=90, va='top', ha='right', alpha=0.5)

    plt.tight_layout()

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"charts/distribution_graph_{timestamp}.png"

    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    
    plt.show()

# Run simulation and plot results

if __name__ == "__main__":
    num_epochs = 8
    epoch_duration = timedelta(days=1)  # Can be changed to any duration
    tokens_to_distribute = 5000
    dist = run_simulation(tokens_to_distribute, num_epochs, epoch_duration)
    plot_distribution(dist)
