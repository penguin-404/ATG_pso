# Generated by Django 3.2.25 on 2024-07-28 16:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SchedulerApp', '0004_alter_meetingtime_day'),
    ]

    operations = [
        migrations.AlterField(
            model_name='meetingtime',
            name='day',
            field=models.CharField(choices=[('Sunday', 'Sunday'), ('Monday', 'Monday'), ('Tuesday', 'Tuesday'), ('Wednesday', 'Wednesday'), ('Thursday', 'Thursday')], max_length=15),
        ),
    ]
