// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package k8s

import (
	"context"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func testJob(name string) *batchv1.Job {
	return &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "slemify"}}
}

// A deploy that returns while a --no-wait training is still running must
// attach to the running Job, not delete and restart it (an attendee's
// in-flight fine-tune was being killed by the returning deploy).
func TestSubmitJobAttachesToRunningJob(t *testing.T) {
	running := testJob("p-training")
	running.Status.Active = 1
	running.UID = "original"
	cs := fake.NewSimpleClientset(running)
	c := &Client{clientset: cs, namespace: "slemify"}

	name, err := c.SubmitJob(context.Background(), testJob("p-training"))
	if err != nil {
		t.Fatal(err)
	}
	if name != "p-training" {
		t.Fatalf("name = %q", name)
	}
	got, _ := cs.BatchV1().Jobs("slemify").Get(context.Background(), "p-training", metav1.GetOptions{})
	if got.UID != "original" {
		t.Fatal("running job was replaced; SubmitJob must attach to it")
	}
}

// A finished job is replaced: re-running a stage explicitly means a fresh run.
func TestSubmitJobReplacesFinishedJob(t *testing.T) {
	done := testJob("p-training")
	done.Status.Succeeded = 1
	done.UID = "original"
	cs := fake.NewSimpleClientset(done)
	c := &Client{clientset: cs, namespace: "slemify"}

	if _, err := c.SubmitJob(context.Background(), testJob("p-training")); err != nil {
		t.Fatal(err)
	}
	got, _ := cs.BatchV1().Jobs("slemify").Get(context.Background(), "p-training", metav1.GetOptions{})
	if got.UID == "original" {
		t.Fatal("finished job was not replaced")
	}
}

// JobSucceeded is true only for a Job with JobComplete=True. The plain deploy
// after a --no-wait training uses it to reuse a finished Job's output.
func TestJobSucceeded(t *testing.T) {
	complete := testJob("p-complete")
	complete.Status.Succeeded = 1
	complete.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}
	running := testJob("p-running")
	running.Status.Active = 1
	failed := testJob("p-failed")
	failed.Status.Failed = 3
	failed.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}
	cs := fake.NewSimpleClientset(complete, running, failed)
	c := &Client{clientset: cs, namespace: "slemify"}

	for name, want := range map[string]bool{
		"p-complete": true,
		"p-running":  false,
		"p-failed":   false,
		"p-missing":  false,
	} {
		got, err := c.JobSucceeded(context.Background(), name)
		if err != nil {
			t.Errorf("%s: unexpected error %v", name, err)
		}
		if got != want {
			t.Errorf("%s: JobSucceeded = %v, want %v", name, got, want)
		}
	}
}

// A failed job is replaced even if pods are still winding down.
func TestSubmitJobReplacesFailedJob(t *testing.T) {
	failed := testJob("p-training")
	failed.Status.Active = 1
	failed.Status.Failed = 1
	failed.UID = "original"
	cs := fake.NewSimpleClientset(failed)
	c := &Client{clientset: cs, namespace: "slemify"}

	if _, err := c.SubmitJob(context.Background(), testJob("p-training")); err != nil {
		t.Fatal(err)
	}
	got, _ := cs.BatchV1().Jobs("slemify").Get(context.Background(), "p-training", metav1.GetOptions{})
	if got.UID == "original" {
		t.Fatal("failed job was not replaced")
	}
}
